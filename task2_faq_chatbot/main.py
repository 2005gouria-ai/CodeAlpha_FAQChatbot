"""
Chatbot for FAQs
================

A retrieval-based chatbot that answers user questions from a fixed FAQ set
using NLP preprocessing (NLTK) and cosine similarity.

Run from the VS Code terminal:

    python main.py            # launches the Tkinter chat GUI
    python main.py --cli      # runs a chat session in the terminal

Requirements: see requirements.txt
"""

import argparse
import os

from chatbot import FAQChatbot


FALLBACK = ("Sorry, I don't have an answer for that yet. "
            "Try asking about the internship, tasks, certificate, "
            "submission, or contact details.")


# --------------------------------------------------------------------------
# Terminal (CLI) mode
# --------------------------------------------------------------------------
def run_cli():
    bot = FAQChatbot()
    print("=" * 60)
    print(f" FAQ Chatbot — {bot.topic}")
    print("=" * 60)
    print("Ask a question, or type 'quit' to exit.")
    print("Type 'faqs' to list the available questions.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if query.lower() in {"faqs", "list"}:
            print("\nAvailable questions:")
            for i, rec in enumerate(bot.records, 1):
                print(f"  {i:>2}. {rec['question']}")
            print()
            continue
        if not query:
            continue

        answer, score, matched = bot.get_response(query)
        if answer is None:
            print(f"Bot: {FALLBACK}\n")
        else:
            print(f"Bot: {answer}")
            print(f"     (matched: \"{matched}\", confidence: {score:.2f})\n")


# --------------------------------------------------------------------------
# Desktop (GUI) mode
# --------------------------------------------------------------------------
def run_gui():
    import tkinter as tk
    from tkinter import ttk
    from tkinter.scrolledtext import ScrolledText

    bot = FAQChatbot()

    root = tk.Tk()
    root.title("FAQ Chatbot")
    root.geometry("720x600")
    root.minsize(600, 480)

    tk.Label(root, text=f"FAQ Chatbot — {bot.topic}",
             font=("Segoe UI", 14, "bold"), pady=8).pack()

    chat = ScrolledText(root, wrap="word", font=("Segoe UI", 11),
                        state="disabled", bg="#ffffff")
    chat.pack(fill="both", expand=True, padx=12, pady=(0, 8))
    chat.tag_config("user", foreground="#0b5394", font=("Segoe UI", 11, "bold"))
    chat.tag_config("bot", foreground="#2b7a0b")
    chat.tag_config("meta", foreground="#888888", font=("Segoe UI", 9, "italic"))

    def add(sender, text, tag):
        chat.configure(state="normal")
        chat.insert("end", f"{sender}: ", tag)
        chat.insert("end", f"{text}\n")
        chat.configure(state="disabled")
        chat.see("end")

    def respond(query):
        if not query:
            return
        add("You", query, "user")
        answer, score, matched = bot.get_response(query)
        if answer is None:
            add("Bot", FALLBACK, "bot")
            chat.configure(state="normal")
            chat.insert("end", "\n")
            chat.configure(state="disabled")
        else:
            add("Bot", answer, "bot")
            chat.configure(state="normal")
            chat.insert("end", f"(confidence: {score:.2f})\n\n", "meta")
            chat.configure(state="disabled")

    entry_var = tk.StringVar()

    def on_send(_event=None):
        respond(entry_var.get().strip())
        entry_var.set("")

    bottom = tk.Frame(root)
    bottom.pack(fill="x", padx=12, pady=(0, 12))
    entry = ttk.Entry(bottom, textvariable=entry_var, font=("Segoe UI", 11))
    entry.pack(side="left", fill="x", expand=True)
    entry.bind("<Return>", on_send)
    ttk.Button(bottom, text="Send", command=on_send).pack(side="left", padx=(8, 0))

    def show_faqs():
        add("Bot", "Here are things you can ask me:", "bot")
        for rec in bot.records:
            add("•", rec["question"], "meta")

    ttk.Button(root, text="Show FAQs", command=show_faqs).pack(pady=(0, 6))

    add("Bot", f"Hi! Ask me anything about the {bot.topic}.", "bot")
    chat.configure(state="normal")
    chat.insert("end", "\n")
    chat.configure(state="disabled")
    entry.focus_set()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Chatbot for FAQs")
    parser.add_argument("--cli", action="store_true",
                        help="run in the terminal instead of the desktop GUI")
    args = parser.parse_args()

    if args.cli:
        run_cli()
        return
    try:
        run_gui()
    except Exception as exc:
        print(f"[warning] Could not start the GUI ({exc}). Falling back to terminal mode.\n")
        run_cli()


if __name__ == "__main__":
    main()
