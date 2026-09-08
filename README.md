# AutoGPT 风格自主智能体 —— 基于 ReAct 范式的智能数据问答系统

## 项目简介

本项目是一个 **基于 ReAct（Reasoning + Acting）范式的自主智能体系统**，借鉴了 AutoGPT 的设计理念，使用 LangChain 框架构建。

系统的核心思路是：让大语言模型（LLM）不再只是"一问一答"，而是能够 **自主地拆解复杂任务、选择合适的工具、逐步执行并反思**，最终完成需要多步推理的数据分析任务。

### 它能做什么？

用户用自然语言提出与业务数据相关的问题，Agent 会自动：

1. **理解任务** → 识别关键概念并拆解为子任务
2. **探查数据** → 自动读取 `data/` 目录下的 Excel、Word、PDF 文件
3. **分析数据** → 生成并执行 Python 代码进行数据计算
4. **生成文档** → 根据分析结果撰写报告
5. **发送邮件** → 将结果通知相关人员
6. **自我反思** → 每一步都会反思推理过程是否完整、准确

### 示例对话

```
🤖：有什么可以帮您？
👨：帮我找出销售额不达标的供应商，给这两家供应商发一封邮件通知此事

>>>>Round: 0<<<<
[思考] 关键概念: 销售额达标标准、供应商销售额...
[行动] 调用 AskDocument 查询销售计划文档中的达标标准...
>>>>Round: 1<<<<
[思考] 已获取达标标准=XX万，接下来查询各供应商销售额...
[行动] 调用 AnalyseExcel 分析销售记录...
...
>>>>Round: N<<<<
[行动] 调用 FINISH，返回最终结果
```

### 示例问题

* 9月份的销售额是多少
* 销售总额最大的产品是什么
* 帮我找出销售额不达标准的供应商
* 给这两家供应商发一封邮件通知此事
* 对比8月和9月销售情况，写一份报告

---

## 项目架构

```
├── Agent/              # 智能体核心模块
│   ├── Action.py       #   动作数据模型（工具名 + 参数）
│   └── ReAct.py        #   ReAct 推理循环（思考→行动→观察）
├── Models/             # 模型工厂模块
│   └── Factory.py      #   LLM / Embedding 模型的统一创建接口
├── Tools/              # 工具集模块
│   ├── FileQATool.py   #   文档问答（PDF/Word → 向量检索 → 回答）
│   ├── ExcelTool.py    #   Excel 文件结构探查
│   ├── PythonTool.py   #   代码生成 + 沙箱执行（分析 Excel 数据）
│   ├── WriterTool.py   #   文档生成
│   ├── EmailTool.py    #   邮件发送
│   ├── FinishTool.py   #   任务结束占位符
│   └── Tools.py        #   工具注册中心（统一导出）
├── Utils/              # 工具类模块
│   ├── PrintUtils.py   #   彩色终端输出
│   └── CallbackHandlers.py  #   LangChain 回调（实时打印思考过程）
├── prompts/            # 提示词模块
│   ├── main/main.txt   #   主 Prompt（定义 Agent 的思维链格式）
│   └── tools/          #   各工具的专用 Prompt
├── data/               # 工作数据目录（Excel、文档等）
├── main.py             # 入口文件（自研 ReAct Agent）
└── main_lc.py          # 入口文件（LangChain 原生 Agent 实现）
```

### 核心工作流

```
用户提问 → Agent 接收任务
  → 循环（最多 20 步）：
      1. LLM 根据 Prompt + 历史记忆 + 短期记忆 进行结构化思考
      2. 输出 Action（工具名 + 参数）
      3. 执行工具，获取观察结果
      4. 将思考过程和观察追加到短期记忆
  → 输出 FINISH + 最终答案
  → 更新对话历史（长期记忆）
```

---

## 快速开始

### 第一步：设置环境变量

把项目中的 `.env` 里面的 `OPENAI_API_KEY` 配置为自己的：

```
OPENAI_API_KEY=sk-xxxx
```

> 项目也支持使用 SiliconFlow 平台的开源模型（如 Qwen2-72B），需在 `.env` 中配置 `SILICONFLOW_API_KEY`。

### 第二步：设置环境变量

控制某些软件包在编译和运行时是否使用本地（原生）库：

```bash
export HNSWLIB_NO_NATIVE=1
```

### 第三步：安装依赖

```bash
pip install -r requirements.txt
```

### 第四步：运行

运行 `main.py` 文件：

```bash
python main.py
```

然后在界面里面输入问题：

```
🤖：有什么可以帮您？
👨：9月份的销售额是多少
>>>>Round: 0<<<<
...
```

> 也可以运行 `main_lc.py`，该文件使用 LangChain 内置的 `create_react_agent` 实现，功能相同但架构不同。

---

## 技术栈

| 技术 | 用途 |
|------|------|
| LangChain 0.1.19 | Agent 框架、Chain 编排、工具管理 |
| OpenAI GPT-4o / GPT-3.5 | 主 LLM（推理与决策） |
| ChromaDB | 向量数据库（文档问答的检索） |
| Pandas / openpyxl | Excel 数据分析 |
| PyMuPDF | PDF 文档加载 |
| PythonREPL | Python 代码沙箱执行 |
| colorama | 终端彩色输出 |

