import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from app.agents.orchestrator import Orchestrator
from app.core.llm_service import AzureOpenAIProvider, OpenAILLMProvider
from config.settings import get_settings


def _build_orchestrator() -> Orchestrator:
    settings = get_settings()
    # Prefer the plain OpenAI key; strip whitespace since pasted secrets often
    # carry a trailing newline, which httpx rejects as a "Connection error".
    openai_key = (settings.openai_api_key or "").strip()
    azure_key = (settings.azure_openai.api_key or "").strip()
    azure_endpoint = (settings.azure_openai.endpoint or "").strip()
    # A plain OpenAI key ("sk-...") saved under AZURE_OPENAI_API_KEY still
    # belongs to api.openai.com, not to an Azure endpoint.
    if not openai_key and azure_key.startswith("sk-"):
        openai_key = azure_key
    print(
        f"[SurfSense] OPENAI_API_KEY set: {bool(settings.openai_api_key)}, "
        f"AZURE_OPENAI_API_KEY set: {bool(azure_key)}, "
        f"AZURE_OPENAI_ENDPOINT: {azure_endpoint!r}",
        flush=True,
    )
    if openai_key:
        print("[SurfSense] Using OpenAI provider", flush=True)
        provider = OpenAILLMProvider(
            api_key=openai_key,
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            max_tokens=settings.azure_openai.max_tokens,
            temperature=settings.azure_openai.temperature,
        )
    elif azure_endpoint.startswith(("http://", "https://")) and azure_key:
        print("[SurfSense] Using Azure OpenAI provider", flush=True)
        provider = AzureOpenAIProvider(
            endpoint=azure_endpoint,
            api_key=azure_key,
            deployment_name=settings.azure_openai.deployment_name,
            api_version=settings.azure_openai.api_version,
            temperature=settings.azure_openai.temperature,
            max_tokens=settings.azure_openai.max_tokens,
        )
    else:
        raise RuntimeError(
            "No usable LLM config. Set the OPENAI_API_KEY secret (sk-...), or set "
            "AZURE_OPENAI_API_KEY plus AZURE_OPENAI_ENDPOINT starting with https://."
        )
    return Orchestrator(provider, settings)


async def _respond(user_msg: str, history: list, orchestrator):
    if not user_msg.strip():
        return history, "", orchestrator
    try:
        if orchestrator is None:
            orchestrator = _build_orchestrator()
        response = await orchestrator.process(user_msg)
    except Exception as e:
        # The OpenAI SDK wraps the real failure (bad header, DNS, proxy, ...) in
        # a generic "Connection error." — surface the underlying cause.
        detail = str(e)
        cause = e.__cause__ or e.__context__
        while cause is not None:
            detail += f" ← {type(cause).__name__}: {cause}"
            cause = cause.__cause__ or cause.__context__
        print(f"[SurfSense] request failed: {type(e).__name__}: {detail}", flush=True)
        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"⚠️ Error: {detail}"},
        ]
        return history, "", orchestrator
    history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": response},
    ]
    return history, "", orchestrator


def _reset(orchestrator):
    if orchestrator is not None:
        orchestrator.reset()
    return [], orchestrator


_css = """
/* ── Universal ─────────────────────────────────────────── */
.gradio-container { max-width: 900px !important; margin: auto; }
.chatbot { border-radius: 12px !important; }
.chatbot .message.user .bubble { background: #6366f1 !important; color: #ffffff !important; }
footer { display: none !important; }
button:has(svg.feather-maximize),
button:has(svg.feather-minimize) { display: none !important; }

/* ── Dark mode ──────────────────────────────────────────── */
/* Gradio 5 toggles class="dark" on document.body,          */
/* and also honours ?__theme=dark used by HF Spaces.        */
body.dark,
body.dark .gradio-container { background: #0d1117 !important; color: #e2e8f0 !important; }
body.dark .chatbot,
body.dark .chatbot .bubble-wrap { background: #0d1117 !important; }
body.dark .chatbot .message.bot .bubble { background: #1e2433 !important; color: #e2e8f0 !important; }
"""

with gr.Blocks(title="SurfSense", theme=gr.themes.Soft(), css=_css) as demo:
    gr.Markdown(
        "# 🏄 SurfSense\n"
        "**Your AI-powered surf trip planning assistant.**\n\n"
        "---\n\n"
        "### What SurfSense does\n"
        "SurfSense combines real-time surf forecasts, spot knowledge, and AI reasoning to help you plan smarter surf sessions and trips. "
        "Tell it where you want to surf and your skill level, and it will:\n\n"
        "- 🌊 **Fetch live forecasts** — wave height, period, swell direction, wind, and tide\n"
        "- 📍 **Research the spot** — break type, hazards, crowd levels, and skill suitability\n"
        "- ✅ **Assess conditions** — rated Good / Fair / Poor for your experience level\n"
        "- ⚠️ **Flag safety concerns** — rip currents, reef exposure, strong winds, and more\n"
        "- 🗓️ **Find the best windows** — pinpoints the optimal days and times over the next week\n"
        "- 🗺️ **Build multi-day itineraries** — schedules sessions across multiple spots for longer trips\n\n"
        "### How to use it\n"
        "Just chat naturally. A good starting message includes your **skill level** (beginner, intermediate, or advanced) "
        "and the **spot or region** you have in mind. For example:\n\n"
        "> *\"I want to surf tomorrow at Ericeira — I'm an intermediate surfer, which spot should I go to?\"*\n\n"
        "> *\"Planning a weekend trip from Lisbon to Peniche, I surf at a beginner level — what are the best spots?\"*\n\n"
        "You can ask follow-up questions, compare spots, request safety details, or refine the itinerary — SurfSense keeps the full conversation context.\n\n"
        "### What to expect\n"
        "Responses typically take **10–30 seconds** as SurfSense fetches live data and runs its analysis pipeline in the background. "
        "You'll get a structured breakdown of conditions, a suitability rating, safety notes, and — for multi-day trips — a day-by-day schedule. "
        "Use the **Reset conversation** button below to start a fresh session for a new destination.\n\n"
        "---"
    )

    state = gr.State(None)

    chatbot = gr.Chatbot(height=520, label="SurfSense")

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="e.g. I want to surf tomorrow at Ericeira — I'm an intermediate surfer, where should I go?",
            show_label=False,
            scale=9,
            lines=1,
        )
        send_btn = gr.Button("Send", scale=1, variant="primary")

    reset_btn = gr.Button("Reset conversation", variant="secondary", size="sm")

    for trigger in (send_btn.click, msg_box.submit):
        trigger(
            fn=_respond,
            inputs=[msg_box, chatbot, state],
            outputs=[chatbot, msg_box, state],
            api_name=False,
        )

    reset_btn.click(fn=_reset, inputs=[state], outputs=[chatbot, state], api_name=False)


if __name__ == "__main__":
    demo.launch()
