from flask import Blueprint, render_template, request, session

from .services.ml_service import get_top_services, run_prediction


web = Blueprint("web", __name__)


def get_chat_history():
    return session.get("chat_history", [])


def save_chat_history(history):
    session["chat_history"] = history[-8:]


def render_home(**kwargs):
    context = {
        "services": get_top_services(),
        "chat_history": get_chat_history(),
        "chat_error": None,
        "chat_question": "",
        "result": None,
        "prediction_explanation": None,
        "responsible_features": [],
        "prediction_confidence": None,
    }
    context.update(kwargs)
    return render_template("index2.html", **context)


@web.route("/")
def home():
    return render_home()


@web.route("/predict", methods=["POST"])
def predict():
    try:
        prediction_result = run_prediction(request.form.to_dict())
        return render_home(**prediction_result)
    except Exception as exc:
        return str(exc)


@web.route("/chat", methods=["POST"])
def chat():
    question = request.form.get("question", "").strip()
    if not question:
        return render_home(chat_error="Enter a question about the NSL-KDD dataset.")

    history = get_chat_history()

    try:
        from chat.genai import ask_pdf

        answer = ask_pdf(question)
    except Exception as exc:
        return render_home(
            chat_error=f"Chat is unavailable right now: {exc}",
            chat_question=question,
        )

    history.append({"role": "user", "message": question})
    history.append({"role": "assistant", "message": answer})
    save_chat_history(history)

    return render_home(chat_question="")
