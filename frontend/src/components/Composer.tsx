import { useEffect, useRef, useState } from 'react';
import { api, newIdempotencyKey } from '../api';
import { useAuth } from '../auth';
import { useToast } from './Toast';

// 與後端 app/models/schemas.py 的 MESSAGE_MAX_CHARS 一致
const MESSAGE_MAX_CHARS = 2000;
const MESSAGE_COUNTER_FROM = 1600;

/**
 * 輸入區。回覆進行中仍可送出（後端排隊），旁邊多一顆停止鈕：
 * 想「打斷再說」就先按停止，再送出新訊息。
 */
export default function Composer({
  onSend, onStop, streaming, stopping,
}: {
  onSend: (text: string, imageUrl: string | null, idempotencyKey: string) => Promise<boolean>;
  onStop: () => void;
  streaming: boolean;
  stopping: boolean;
}) {
  const [text, setText] = useState('');
  const [image, setImage] = useState<{ url: string; preview: string } | null>(null);
  const [uploading, setUploading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const failedSend = useRef<{ text: string; imageUrl: string | null; key: string } | null>(null);
  const toast = useToast();
  const { status } = useAuth();

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
  }, [text]);

  const length = text.trim().length;
  const tooLong = length > MESSAGE_MAX_CHARS;
  const canSend = (length > 0 || !!image) && !uploading && !tooLong;

  async function submit() {
    if (!canSend) return;
    const t = text.trim();
    const img = image;
    const imageUrl = img?.url ?? null;
    // 送出失敗時伺服器可能其實收到了（只是回應遺失）：內容沒改就是重送同一次，沿用 key，
    // 不會多跑一個任務。改了內容，或清掉重打但文字不同，都算新訊息。
    const failed = failedSend.current;
    const key = failed && failed.text === t && failed.imageUrl === imageUrl ? failed.key : newIdempotencyKey();
    failedSend.current = null;
    setText('');
    setImage(null);
    const ok = await onSend(t, imageUrl, key);
    if (!ok) {
      failedSend.current = { text: t, imageUrl, key };
      setText(t);
      setImage(img);
    }
    textareaRef.current?.focus();
  }

  async function pickImage(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    if (!file.type.startsWith('image/')) return toast('請上傳圖片檔案', true);
    // 上限與後端一致：訪客圖片轉 Base64 送給模型（Groq 對 Base64 請求上限 4MB）→ 2.5MB；登入使用者上傳到 GCS → 10MB
    const limitMb = status === 'user' ? 10 : 2.5;
    if (file.size > limitMb * 1024 * 1024) return toast(`圖片大小不可超過 ${limitMb}MB`, true);
    setUploading(true);
    try {
      if (status !== 'user') {
        // 訪客不留任何檔案：圖片轉成 Base64 直接放進這則訊息，不上傳到雲端
        setImage({ url: await api.imageToDataUrl(file), preview: URL.createObjectURL(file) });
        return;
      }
      const ext = file.name.split('.').pop() || 'jpg';
      const presign = await api.presign(`${Date.now()}_${Math.random().toString(36).slice(2, 8)}.${ext}`);
      toast('正在上傳圖片至雲端...');
      // uploadHeaders（Content-Type、檔案大小範圍）都在簽章裡，要原樣帶上
      const up = await fetch(presign.uploadUrl, { method: 'PUT', headers: presign.uploadHeaders, body: file });
      if (!up.ok) throw new Error(`GCS 上傳失敗: HTTP ${up.status}`);
      setImage({ url: presign.accessUrl, preview: URL.createObjectURL(file) });
      toast('圖片已上傳至雲端 ☁️');
    } catch (err) {
      toast(`圖片上傳失敗: ${(err as Error).message}`, true);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="input-area">
      <div className="input-wrapper">
        <div className={`image-preview-container${image ? ' has-image' : ''}`}>
          {image && <img className="image-preview" src={image.preview} alt="preview" />}
          <button type="button" className="image-preview-remove" onClick={() => setImage(null)}>✕</button>
        </div>
        <div className="input-box">
          <textarea
            ref={textareaRef}
            data-testid="message-input"
            placeholder={streaming ? '回覆中也可以繼續輸入，送出後會排隊… (Ctrl+Enter 送出)' : '輸入訊息... (Ctrl+Enter 送出)'}
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                submit();
              }
            }}
          />
          <div className="input-actions">
            <button type="button" className="input-btn" onClick={() => fileRef.current?.click()} title="上傳圖片" disabled={uploading}>
              {uploading ? <span className="spinner-small" /> : '📎'}
            </button>
            {streaming && (
              <button
                type="button"
                className="stop-btn"
                data-testid="stop-button"
                onClick={onStop}
                disabled={stopping}
                title="停止回覆"
              >
                {stopping ? <span className="spinner-small" /> : '■'}
              </button>
            )}
            <button type="button" className="send-btn" data-testid="send-button" onClick={submit} disabled={!canSend} title="送出">
              ➤
            </button>
          </div>
        </div>
        {/* 不用 textarea 的 maxLength：它會默默截斷貼上的文字，使用者不會發現後面不見了 */}
        <div className={`input-hint${tooLong ? ' is-over-limit' : ''}`} data-testid="input-hint">
          {tooLong
            ? `訊息太長（${length} / ${MESSAGE_MAX_CHARS} 字），請縮短後再送出`
            : length >= MESSAGE_COUNTER_FROM
              ? `${length} / ${MESSAGE_MAX_CHARS} 字`
              : '按 Ctrl + Enter 送出 · 回覆中可繼續送出（自動排隊）或按 ■ 停止 · 支援圖片上傳'}
        </div>
      </div>
      <input type="file" ref={fileRef} accept="image/*" onChange={pickImage} style={{ display: 'none' }} />
    </div>
  );
}
