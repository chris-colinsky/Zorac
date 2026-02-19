# Learn AI Engineering with Real Hardware

**Build a self-hosted LLM chat client and learn the full stack — from GPU inference to terminal UI.**

Zorac is an educational open-source project that teaches AI engineering concepts by building something real: a ChatGPT-style chat client that runs entirely on your own hardware. No cloud APIs, no monthly costs, complete privacy.

This documentation site goes beyond "how to install" and explains the **why** behind every design decision — so you can apply these patterns to your own projects.

---

## What You'll Learn

<div class="grid cards" markdown>

-   **Concepts**

    ---

    Understand the fundamentals: how LLMs generate text, why quantization lets you run 24B-parameter models on a gaming GPU, how inference servers work, and how to manage context windows.

    [:octicons-arrow-right-24: Start with Concepts](site/concepts/how-llms-work.md)

-   **Guides**

    ---

    Step-by-step guides for building each component: setting up a vLLM inference server, building a terminal UI with Textual, and configuring multi-GPU training.

    [:octicons-arrow-right-24: Browse Guides](site/guides/building-the-tui.md)

-   **Walkthroughs**

    ---

    Trace through the actual source code to see how everything connects. Follow a message from keypress to rendered response, or understand how streaming markdown works.

    [:octicons-arrow-right-24: Read Walkthroughs](site/walkthroughs/what-happens-on-enter.md)

-   **Decisions**

    ---

    Architecture Decision Records explaining _why_ we chose Textual over other TUI frameworks, AWQ over other quantization formats, and other key trade-offs.

    [:octicons-arrow-right-24: See Decisions](site/decisions/why-textual.md)

</div>

---

## Who This Is For

- **Developers** who want to understand how local LLM applications work end-to-end
- **AI engineers** looking to run inference on consumer hardware without cloud dependencies
- **Students** learning about quantization, tokenization, and context management
- **Homelab enthusiasts** who want to self-host their own ChatGPT alternative
- **Anyone with a gaming GPU** (RTX 3080 or better) curious about running AI locally

---

## Quick Links

| Getting Started | Reference |
|----------------|-----------|
| [Install Zorac](INSTALLATION.md) | [Configuration Reference](CONFIGURATION.md) |
| [Set up a vLLM Server](SERVER_SETUP.md) | [Usage & Commands](USAGE.md) |
| [Understand Quantization](site/concepts/quantization.md) | [Development Guide](DEVELOPMENT.md) |
| [What Happens When You Press Enter](site/walkthroughs/what-happens-on-enter.md) | [GitHub Repository](https://github.com/chris-colinsky/Zorac) |
