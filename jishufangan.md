# 校园教务 RAG Agent — 技术方案文档

**版本**：v1.0  
**日期**：2026-09-11  
**项目定位**：面向校园教务场景的本地化 RAG 问答 Agent，覆盖学分、专业设置、转专业、学士学位四个主题  
**目标**：跑通完整 RAG 链路，面试时可展示工程决策与取舍


## 一、项目概述

### 1.1 问题定义

校园教务信息分散在通知、手册、网页中，学生查询效率低。传统方案要么靠人工客服，要么靠关键词搜索，都存在“不知道去哪找”“找到的看不懂”的问题。

本项目构建一个**本地部署的教务 RAG Agent**，学生用自然语言提问，Agent 从知识库检索相关片段，基于真实资料生成回答，并附来源引用。

### 1.2 核心能力

- 回答学分要求、专业设置、转专业条件、学士学位申请等教务问题
- 回答附来源引用（来自哪个文档）
- 检索不到时明确拒答，不编造
- 支持多轮对话（Streamlit session_state）

### 1.3 非目标（明确不做）

- 不做用户系统、不做权限管理（MVP 阶段）
- 不做全校数据爬取（手工整理 4 个文档）
- 不做多 Agent 编排（单 Agent + RAG + 1 个工具）


## 二、框架选型

### 2.1 核心决策：LangChain 还是 LlamaIndex？

这是本方案最关键的选型决策。2026 年的主流判断标准是：**看你的“难点”在哪。**

- **LlamaIndex**：当核心挑战是“从私有数据中高质量检索”时选择。文档索引、切分、检索管线更精致，RAG 场景代码量更少，检索精度可达约 92%。
- **LangChain**：当任务需要**编排多步骤 Agent 和工具调用**时选择。Agent 抽象成熟，生态集成最广。

**本项目的选择：LangChain。**

理由：本项目虽然当前以 RAG 为主，但 Day5 要加入“课表查询工具调用”，Agent 需要**自主决定**走 RAG 还是走工具。这是编排问题，不是纯检索问题。LangChain 的 `@tool` 装饰器和 Agent 路由机制更适合这个需求。

> 补充：LangChain 也提供了对 LlamaIndex retriever 的一等集成。如果后续检索质量成为瓶颈，可以单独把 LlamaIndex 的检索器接进来，不必推翻整个框架。

### 2.2 备选方案对比

| 维度 | LangChain | LlamaIndex | 自研（纯 Python） |
|---|---|---|---|
| RAG 检索精度 | 良好 | 优秀 | 取决于实现 |
| Agent/工具编排 | 最成熟 | 有 Workflows，但生态较新 | 需自己写 |
| 学习曲线 | 较陡 | RAG 场景较低 | 陡（什么都要自己写） |
| 社区与文档 | 最丰富 | 丰富 | 无 |
| 本项目适配度 | ✅ 首选 | 备选 | 不推荐 |


## 三、技术栈

### 3.1 最终技术栈

| 层级 | 选型 | 版本/规格 | 理由 |
|---|---|---|---|
| **编排框架** | LangChain | 0.3.x | Agent + RAG 生态最成熟 |
| **LLM** | deepseek-r1:8b (Ollama) | 已部署 | 本地免费，中文效果好，无需 API Key |
| **Embedding** | BAAI/bge-small-zh-v1.5 | ~80MB | 中文 RAG 首选小模型，速度极快 |
| **向量库** | ChromaDB | 最新稳定版 | 零配置启动，与 LangChain 深度集成 |
| **文档切分** | RecursiveCharacterTextSplitter | LangChain 内置 | 按标点/段落自然切分，避免硬切 |
| **前端** | Streamlit | 最新稳定版 | 快速出 Demo，Day1 已验证可用 |
| **部署** | Streamlit Community Cloud | 免费 | GitHub 直连，一键部署 |
| **版本管理** | Git + GitHub | 已配置 | Day1 已完成 |

### 3.2 为什么选 Chroma 而不是 Milvus？

Chroma 的定位是“开发者的第一选择”，安装简单、代码量少、零配置启动，与 LangChain 深度集成。它的局限是大规模检索性能下降明显，但对于本项目 4 个文档、几十个片段的规模，完全够用。

Milvus 适合百万级向量、生产环境集群部署。本项目不需要。如果未来数据量增长到需要 Milvus 的规模，迁移成本也可控（LangChain 的 vectorstore 接口统一）。

### 3.3 为什么选 bge-small-zh-v1.5？

三个关键参数：

- **模型大小**：仅约 80MB，CPU 即可运行，不需要 GPU
- **速度**：⚡⚡⚡⚡⚡（最高等级），中文场景速度极快
- **Token 上限**：512 tokens，中文约 1 字符 ≈ 1 token，所以每个 chunk 控制在 400-500 字符以内

BGE 系列是北京智源研究院推出的中文 Embedding 模型，在中文 RAG 场景中效果最好。small 版本在“速度 + 体积 + 效果”三者间平衡最佳，适合个人项目。


## 四、系统架构

### 4.1 整体链路

系统分为**离线索引链路**和**在线检索生成链路**两条。

**离线索引（构建一次）：**

```
docs/ 下的 4 个 Markdown 文件
  → 文档加载（DirectoryLoader）
  → 文本切分（RecursiveCharacterTextSplitter, chunk_size=500, overlap=80）
  → 向量化（bge-small-zh-v1.5）
  → 存入 ChromaDB
```

**在线检索生成（每次提问）：**

```
用户提问
  → Streamlit 输入框
  → Agent 判断问题类型
    ├─ 教务知识问题 → Chroma 检索 Top3 → 拼入 Prompt → DeepSeek 生成回答 + 来源
    ├─ 课表查询 → SQLite 工具调用 → DeepSeek 整理回答
    └─ 无关/敏感问题 → 拒答
  → 返回回答到前端
```

### 4.2 Agent 的 ReAct 循环

本项目采用 **ReAct（Reason + Act）模式**。Agent 先“思考”问题需要什么信息，再“行动”调用相应工具，观察结果后决定是否继续。

```
Thought: 用户问“转专业需要什么条件”，这是知识型问题，需要查知识库。
Action: search_knowledge_base("转专业 申请条件")
Observation: [检索到 3 个相关片段...]
Thought: 信息足够，可以回答。
Final Answer: 基于检索到的片段生成回答，附来源。
```

ReAct 的优势是**可审计**：前端可以展示“正在判断 → 准备调用工具 → 工具返回结果 → 准备回答”的步骤。

### 4.3 工具设计

MVP 阶段设计 **2 个工具**：

| 工具名 | 功能 | 触发场景 |
|---|---|---|
| `search_knowledge_base` | 从 Chroma 检索教务知识 | 学分/专业/转专业/学位问题 |
| `query_schedule` | 查询 SQLite 中的课表数据 | “我这学期有什么课” |

Agent 通过 Function Calling 自主决定调用哪个工具，不需要用户指定。


## 五、关键实现细节

### 5.1 文档切分策略

| 参数 | 值 | 理由 |
|---|---|---|
| chunk_size | 500 字符 | bge-small-zh 上限 512 token，留余量 |
| chunk_overlap | 80 字符 | 避免信息被割裂，约 15% 重叠 |
| 切分器 | RecursiveCharacterTextSplitter | 按 `\n\n` → `\n` → 句号 → 逗号 逐级切分 |

**常见坑**：不要按固定字数硬切。硬切会把“申请条件”和“不允许转专业的情况”切散，导致检索不准。

### 5.2 拒答机制

当检索到的片段与问题的相关性低于阈值时，Agent 返回“这个问题我暂时无法回答，建议咨询教务处”，而不是让模型编造。

**实现方式**：计算检索结果的向量相似度分数，低于阈值（如 0.5）则触发拒答。

### 5.3 提示词设计（Day3 实现）

```
你是一个校园教务助手。请严格基于以下参考资料回答学生问题。

规则：
1. 只使用参考资料中的信息，不要编造
2. 如果参考资料不足以回答问题，明确说“根据现有资料无法回答”
3. 回答末尾附上来源文件名
4. 用简洁中文回答，不要超过 200 字

参考资料：
{context}

学生问题：{question}
```


## 六、评估方案

### 6.1 核心评估指标

| 指标 | 含义 | 目标值 |
|---|---|---|
| **检索命中率** | Top3 片段中包含正确答案的比例 | ≥ 85% |
| **Faithfulness** | 生成的回答中，每个断言都能被检索到的上下文支持的比例 | ≥ 90% |
| **拒答准确率** | 无关问题正确拒答的比例 | ≥ 80% |
| **响应时间** | 从提问到回答的总耗时 | ≤ 10 秒 |

### 6.2 测试集设计

准备 **20 个测试问题**：

- 10 个知识型问题（覆盖 4 个文档，每个 2-3 题）
- 5 个课表查询问题
- 5 个无关/敏感问题（测试拒答）

Day7 逐一测试，记录命中率和 Faithfulness。

### 6.3 调优手段

如果指标不达标，按顺序尝试：

1. 调整 chunk_size（500 → 300 或 800）
2. 调整 overlap（80 → 50 或 120）
3. 调整检索 Top-K（3 → 5）
4. 改用混合检索（向量 + BM25 关键词）
5. 最后才考虑换 Embedding 模型（bge-small → bge-base）


## 七、文件结构

```
campus-agent/
  docs/
    学分.md
    专业设置.md
    转专业.md
    学士学位.md
  data/
    schedule.db          # SQLite 课表数据
  app.py                 # Streamlit 界面
  agent.py               # Agent 路由与 ReAct 循环
  retriever.py           # Chroma 检索逻辑
  indexer.py             # 离线索引：文档加载 → 切分 → 嵌入 → 入库
  tools.py               # 工具定义（知识库检索、课表查询）
  requirements.txt
  README.md
  .gitignore
  notes.md
```

### 依赖清单

```txt
streamlit
langchain
langchain-community
langchain-ollama
langchain-chroma
chromadb
sentence-transformers
```


## 八、开发路线图

| 阶段 | 内容 | 产出 |
|---|---|---|
| Day1 ✅ | 环境搭建 + Streamlit 聊天框 + Ollama 接入 | 能跑通界面→模型→回答 |
| **Day2** | 文档切分 + Chroma 向量库 + 检索函数 | 输入问题返回 Top3 片段 |
| **Day3** | RAG 问答 + 来源引用 + 提示词设计 | 能基于知识库回答 |
| **Day4** | 拒答机制 + 敏感词过滤 + 边界处理 | 无关问题不乱答 |
| **Day5** | 课表查询工具 + Agent 路由 | 项目从 RAG 升级为 Agent |
| **Day6** | 部署到 Streamlit Cloud + README | 公网 URL + 完整文档 |
| **Day7** | 20 题测试 + 调优 + Demo 录制 + 简历 | 测试报告 + 简历话术 |


## 九、面试要点（简历与问答）

### 9.1 简历描述模板

> 独立开发校园教务 RAG Agent，基于 **LangChain + Chroma + bge-small-zh** 构建检索增强问答，支持来源引用与拒答机制；集成 SQLite 课表查询工具，实现 **ReAct 模式**下的 RAG 与工具调用混合路由；部署至 Streamlit Cloud，20 个测试问题检索命中率达 XX%。

### 9.2 面试高频问题准备

按“**业务问题 → Agent 链路 → 关键模块 → 指标与结果**”的顺序组织回答。

**Q：为什么选 LangChain 而不是 LlamaIndex？**  
→ 因为项目需要 Agent 自主决定走 RAG 还是走工具，这是编排问题。LlamaIndex 在纯检索场景更优，但 LangChain 的 Agent 抽象更成熟。

**Q：Agent 在学术上由哪些部分组成？**  
→ **Planning（规划）+ Memory（记忆）+ Tool Use（工具使用）** 三大核心组件。

**Q：怎么减少幻觉？**  
→ 三层防线：（1）提示词约束“只基于参考资料回答”；（2）检索相关性阈值触发拒答；（3）Faithfulness 评估检查回答是否有上下文支持。

**Q：ReAct 循环的完整流程？**  
→ Thought → Action → Observation → 循环直到 Final Answer。每步可审计，前端可展示。


## 十、风险与降级方案

| 风险 | 概率 | 降级方案 |
|---|---|---|
| 检索不准 | 中 | 调整 chunk 参数 → 加 BM25 混合检索 |
| 工具调用卡住 | 中 | 降级为关键词触发（Day5 保底） |
| 部署失败 | 低 | 先 GitHub + 本地录屏，公网后补 |
| 8B 模型太慢 | 低 | 换更小的模型或调低 max_tokens |
| Python 3.14 兼容性 | 低 | 换 Python 3.11/3.12 |


**总结**：本方案的核心决策是**LangChain 做编排 + Chroma 做向量存储 + bge-small-zh 做嵌入 + Ollama 做本地推理**。技术栈全部本地可运行，零 API 成本，架构清晰可讲。Day2 开始按路线图执行即可。