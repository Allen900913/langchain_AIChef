"""背景反思寫入偏好檔案時的保護。

反思（ReflectionExecutor / ReflectingSummarization）是模型在背景自己判斷要改哪些偏好，
使用者看不到也沒有審核，所以從模型輸出到寫進 store 之間加了幾道檢查。

「PatchDoc 被存成新記憶」的成因（2026-10-17 錄到完整的呼叫序列）：
  a. gpt-oss 的 harmony 格式裡工具放在 functions 命名空間，Groq 偶爾沒把前綴去掉，
     回傳的工具名稱是「functions.PatchDoc」。
  b. trustcall 套用 patch 的步驟只認得「PatchDoc」，這筆被跳過，沒有套用。
  c. 驗證步驟回報「工具名稱不存在」，模型用 PatchFunctionName 把名稱改回 PatchDoc，
     但套用 patch 的步驟已經過了；最後 PatchDoc 本身也是合法的 schema，就被當成新記錄交給
     LangMem 存起來：原本的偏好檔沒改到，還多一筆內容是 json_doc_id / patches 的垃圾記錄。
所以有兩道：
  ToolNameFixingChatGroq：在模型輸出時就去掉「functions.」前綴，從源頭避免（a）。
  ProfileGuardStore：萬一還是有 PatchDoc 走到 store，把它套用到原本的偏好檔案上。

ProfileGuardStore 依序做三件事：
1. 修正誤存的 patch：如上。
2. 白名單驗證：只放行「kind 等於該領域的 schema 名稱、欄位都在 schema 裡、型別驗得過」的記錄，
   其他一律丟掉。上面那種 PatchDoc 只是已知的一種壞格式，模型還可能寫出別的；
   與其一種一種列黑名單，不如只認得正確的格式。注意：只管格式，不管內容對不對
   （例如把一次性的 15 分鐘寫進 time_budget 格式完全合法），內容由 evals/reflection_eval.py 把關。
3. 過敏原只增不減：寫錯的代價不對稱，多記一個只是少推薦幾道菜，
   少記一個會讓 DietarySafetyGuard 以為沒有過敏、推薦出危險的菜。
   要移除必須由使用者在對話中明確要求，走 diet_profile_manage 工具（update / delete 都會進人工審核），
   那條路徑用 mode="agent"，不做這項保護。

做法：把 store 包一層，攔截寫進偏好 namespace 的 put / delete。
BaseStore 的 put / aput / delete / adelete 最後都會進 batch / abatch，所以只要改寫這兩個。

主 agent 的 *_profile_manage 工具也寫同一批偏好檔，所以 agent 用的 store 也包一層（mode="agent"）：
- 這些工具（LangMem create_manage_memory_tool）只存 {"content": ...}、沒有 kind，
  而反思讀舊記錄時固定讀 value["kind"]，缺了就 KeyError，該使用者這個領域的反思從此每次失敗。
  kind 由 namespace 唯一決定，所以直接補上（不需要模型判斷的錯誤，程式修）。
- 白名單不合格時拋 ValueError：工具設了 handle_tool_error，錯誤會回給模型重寫，
  不像背景反思那樣靜靜丟掉（主 agent 知道自己想存什麼，交給它修）。
- 不做過敏原保護：這條路徑是使用者在對話中明確要求的變更，update / delete 另有人工審核。
"""
from __future__ import annotations

import copy
import uuid
from collections.abc import Iterable
from collections import Counter
from typing import Any, Literal

import jsonpatch
from langchain_core.outputs import ChatResult
from langchain_groq import ChatGroq
from langgraph.store.base import BaseStore, GetOp, Op, PutOp, Result, SearchOp
from pydantic import BaseModel, ValidationError

_HARMONY_TOOL_PREFIX = "functions."


class ToolNameFixingChatGroq(ChatGroq):
    """去掉工具名稱上 gpt-oss harmony 格式殘留的「functions.」前綴（成因見模組說明 a）。

    LangMem 只接受 BaseChatModel 實例（包成 RunnableLambda 會被拒），所以用子類別改寫輸出。
    """

    @staticmethod
    def _fix_tool_names(result: ChatResult) -> ChatResult:
        for gen in result.generations:
            msg = gen.message
            for tc in getattr(msg, "tool_calls", None) or []:
                tc["name"] = tc["name"].removeprefix(_HARMONY_TOOL_PREFIX)
            # 原始格式也一併改，模型重試時送回去的歷史才一致
            for raw in msg.additional_kwargs.get("tool_calls") or []:
                fn = raw.get("function") or {}
                if isinstance(fn.get("name"), str):
                    fn["name"] = fn["name"].removeprefix(_HARMONY_TOOL_PREFIX)
        return result

    def _generate(self, *args, **kwargs) -> ChatResult:
        return self._fix_tool_names(super()._generate(*args, **kwargs))

    async def _agenerate(self, *args, **kwargs) -> ChatResult:
        return self._fix_tool_names(await super()._agenerate(*args, **kwargs))


def _is_patch(value: dict | None) -> bool:
    content = (value or {}).get("content")
    return isinstance(content, dict) and "patches" in content and "json_doc_id" in content


def _doc_id(namespace: tuple[str, ...], key: str) -> str:
    # 與 LangMem MemoryStoreManager 給既有記錄編的 id 相同（extraction.py 的 _stable_id）
    return uuid.uuid5(uuid.NAMESPACE_DNS, str((*namespace, key))).hex


def apply_patch_record(patch_value: dict, targets: list[Any]) -> tuple[str, dict] | None:
    """把誤存的 PatchDoc 套到它指向的記錄上，回傳 (key, 新 value)；找不到或套不上回傳 None。

    targets：同 namespace 的既有記錄（Item）。json_doc_id 對不上但只有一筆時，就套在那一筆
    （每個領域只有一份偏好檔案）。
    """
    content = patch_value["content"]
    ns_items = [t for t in targets if not _is_patch(t.value)]
    target = next((t for t in ns_items if _doc_id(t.namespace, t.key) == content["json_doc_id"]), None)
    if target is None and len(ns_items) == 1:
        target = ns_items[0]
    patches = content.get("patches")
    if target is None or not isinstance(patches, list):
        return None
    # remove 不需要 value，模型常附上 "value": null，留著無害但 jsonpatch 對其他多餘欄位較挑剔
    ops = [{k: v for k, v in p.items() if not (p.get("op") == "remove" and k == "value")}
           for p in patches if isinstance(p, dict)]
    try:
        patched = jsonpatch.apply_patch(copy.deepcopy(target.value["content"]), ops)
    except (jsonpatch.JsonPatchException, jsonpatch.JsonPointerException, TypeError, KeyError):
        return None
    # 沿用原記錄的 kind；原記錄若是主 agent 早期寫的（沒有 kind），之後由 _add_kind 補上
    return target.key, {**target.value, "content": patched}


def validate_profile(value: dict, schema: type[BaseModel]) -> tuple[dict | None, str]:
    """白名單：回傳 (正規化後的 value, 原因)；不合格時 value 為 None。

    多餘欄位直接拒絕而不是略過：pydantic 預設會忽略多餘欄位、其餘補預設值，
    像 PatchDoc 那種內容會被「驗過」變成一份全空的偏好，等於把使用者的檔案清空。
    """
    kind, content = value.get("kind"), value.get("content")
    if kind != schema.__name__:
        return None, f"kind 是 {kind!r}，應為 {schema.__name__}"
    if not isinstance(content, dict):
        return None, "content 不是物件"
    unknown = set(content) - set(schema.model_fields)
    if unknown:
        return None, f"有 schema 以外的欄位 {sorted(unknown)}"
    try:
        normalized = schema.model_validate(content).model_dump(mode="json")
    except ValidationError as exc:
        err = exc.errors()[0]
        where = ".".join(str(p) for p in err["loc"])
        # 長度上限見 tools.py 的 PROFILE_ITEM_MAX：講清楚哪一欄、要怎麼改，模型才知道怎麼重寫
        if err["type"] == "profile_item_too_long":
            return None, (f"{where} 太長（上限 {err['ctx']['max_length']} 字）。"
                          "偏好的每一項都要是短詞，例如：花生、清淡、氣炸鍋；不要寫成句子或規則")
        if err["type"] == "profile_too_many_items":
            return None, f"{where} 項目太多（上限 {err['ctx']['max_length']} 項）"
        return None, f"型別不符：{err['msg']}"
    return {"kind": kind, "content": normalized}, ""


def preserve_allergies(old_value: dict | None, new_value: dict | None) -> dict | None:
    """回傳實際要寫入的值：舊值有、新值沒有的過敏原補回去。

    new_value 為 None 代表刪除；舊記錄有過敏原時改成「只保留過敏原」的記錄，不整筆刪掉。
    """
    old_content = (old_value or {}).get("content") or {}
    old_allergies = list(old_content.get("allergies") or [])
    if not old_allergies:
        return new_value
    if new_value is None:
        # 其他欄位清空（list → []、其餘 → None），保持跟 DietProfile 一樣的欄位形狀
        emptied = {k: [] if isinstance(v, list) else None for k, v in old_content.items()}
        return {"kind": old_value["kind"], "content": {**emptied, "allergies": old_allergies}}

    content = dict(new_value.get("content") or {})
    allergies = list(content.get("allergies") or [])
    allergies += [a for a in old_allergies if a not in allergies]
    content["allergies"] = allergies
    return {**new_value, "content": content}


class ProfileGuardStore(BaseStore):
    """包住真正的 store；只有偏好 namespace 的寫入會被檢查，其餘原樣轉發。

    schemas：領域 → schema（tools._PROFILE_DOMAINS）。從外面傳入，這個模組不必 import tools。
    mode：reflection = 背景反思（不合格丟掉、過敏原只增不減）；agent = 主 agent 的工具（不合格拋錯、不保護過敏原）。
    stats：各種保護觸發的次數（kind_added / patch_repaired / patch_dropped / rejected / allergy_kept），給監控與評測用。
    """

    def __init__(self, inner: BaseStore, schemas: dict[str, type[BaseModel]],
                 mode: Literal["reflection", "agent"] = "reflection"):
        self._inner = inner
        self._schemas = dict(schemas)
        self._mode = mode
        self._source = "反思" if mode == "reflection" else "主 agent"
        self.supports_ttl = inner.supports_ttl
        self.stats: Counter[str] = Counter()

    def _domain(self, namespace: tuple[str, ...]) -> str | None:
        # namespace 形狀：("memories", user_id, domain)，見 tools._profile_ns
        if len(namespace) == 3 and namespace[0] == "memories" and namespace[2] in self._schemas:
            return namespace[2]
        return None

    def _where(self, op: Op) -> str:
        return f"{op.namespace[1]}/{op.namespace[2]}"

    def _repair(self, op: PutOp, targets: list[Any]) -> PutOp | None:
        """誤存的 patch → 套用後改寫成對原記錄的更新；套不上就丟掉這筆寫入（回傳 None）。"""
        repaired = apply_patch_record(op.value, targets)
        if repaired is None:
            self.stats["patch_dropped"] += 1
            print(f"🩹 [PROFILE GUARD] 反思輸出了無法套用的 patch，已丟棄：{self._where(op)}")
            return None
        key, value = repaired
        self.stats["patch_repaired"] += 1
        print(f"🩹 [PROFILE GUARD] 反思的 patch 被誤存成新記錄，已改套用到原檔案：{self._where(op)}")
        return PutOp(op.namespace, key, value, op.index, op.ttl)

    def _add_kind(self, op: PutOp, domain: str) -> PutOp:
        if "kind" in op.value:
            return op
        self.stats["kind_added"] += 1
        return op._replace(value={"kind": self._schemas[domain].__name__, **op.value})

    def _validate(self, op: PutOp, domain: str) -> PutOp | None:
        value, reason = validate_profile(op.value, self._schemas[domain])
        if value is None:
            self.stats["rejected"] += 1
            if self._mode == "agent":
                raise ValueError(f"偏好格式不正確，沒有儲存：{reason}。請依照工具參數的欄位重新呼叫。")
            print(f"🚫 [PROFILE GUARD] {self._source}寫入的格式不合 schema，已丟棄：{self._where(op)}（{reason}）")
            return None
        return op._replace(value=value)

    def _guard_allergies(self, op: PutOp, old: Any) -> PutOp:
        value = preserve_allergies(old.value if old else None, op.value)
        if value is op.value:
            return op
        self.stats["allergy_kept"] += 1
        print(f"🛡️ [PROFILE GUARD] 反思想移除過敏原，已保留：{op.namespace[1]}")
        return op._replace(value=value)

    # 單筆寫入的處理順序：補 kind → 修 patch → 白名單 → 過敏原（只有 reflection 模式）。
    # 拆成「要讀什麼」（_read_for）與「讀到之後怎麼處理」（_check），同步 / 非同步共用同一份邏輯。
    def _read_for(self, op: Op) -> Op | None:
        if not isinstance(op, PutOp) or (domain := self._domain(op.namespace)) is None:
            return None
        if op.value is not None and _is_patch(op.value):
            return SearchOp(op.namespace, limit=20)  # 找 patch 指向的記錄；它的舊值也從這裡拿
        if domain == "diet" and self._mode == "reflection":
            return GetOp(op.namespace, op.key)       # 過敏原保護要比對舊值
        return None

    def _check(self, op: Op, fetched: Any) -> Op | None:
        if not isinstance(op, PutOp) or (domain := self._domain(op.namespace)) is None:
            return op
        old = fetched
        if op.value is not None:  # None = 刪除，只需要過敏原保護
            if _is_patch(op.value):
                op = self._repair(op, fetched)
                if op is None:
                    return None
                old = next((t for t in fetched if t.key == op.key), None)
            op = self._validate(self._add_kind(op, domain), domain)
            if op is None:
                return None
        if domain == "diet" and self._mode == "reflection":
            op = self._guard_allergies(op, old)
        return op

    @staticmethod
    def _assemble(ops: list[Op | None], inner_results: list[Result]) -> list[Result]:
        # 被丟掉的寫入在結果裡補 None（PutOp 的結果本來就是 None），維持與輸入一一對應
        it = iter(inner_results)
        return [None if op is None else next(it) for op in ops]

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        checked = []
        for op in ops:
            read = self._read_for(op)
            checked.append(self._check(op, self._inner.batch([read])[0] if read else None))
        return self._assemble(checked, self._inner.batch([op for op in checked if op is not None]))

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        checked = []
        for op in ops:
            read = self._read_for(op)
            checked.append(self._check(op, (await self._inner.abatch([read]))[0] if read else None))
        return self._assemble(checked, await self._inner.abatch([op for op in checked if op is not None]))
