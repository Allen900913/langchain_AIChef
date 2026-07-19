"""私廚 agent 的評測套件（離線 eval，不影響 runtime）。

包含五項指標：
- Faithfulness / Answer Relevancy / Context Relevance  → ragas（judge=NVIDIA llama-3.3-70b）
- Tool Selection Accuracy / 過敏原違反率                  → 自製規則式指標

judge / embeddings 全程走 NVIDIA（同一把 NVIDA_API_KEY），但刻意不用受測的 Qwen
本身當 judge，避免 self-judging bias。
"""
