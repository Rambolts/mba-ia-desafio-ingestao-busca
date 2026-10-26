import psycopg
from openai import APIError

from search import load_settings, search_prompt


def main() -> None:
    """Converse no terminal sobre o documento ingerido no banco vetorial."""
    try:
        load_settings()
    except (ValueError, OSError) as error:
        print(f"Não foi possível iniciar o chat: {error}")
        raise SystemExit(1) from None

    print("Chat sobre o documento. Digite 'sair' ou 'exit' para encerrar.")
    while True:
        try:
            question = input("\nVocê: ").strip()
            if question.lower() in {"sair", "exit"}:
                break
            if not question:
                continue

            answer = search_prompt(question)
            print(f"\nAssistente: {answer}")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        except (ValueError, OSError, psycopg.Error, APIError) as error:
            print(f"Não foi possível responder: {error}")

    print("Chat encerrado.")


if __name__ == "__main__":
    main()
