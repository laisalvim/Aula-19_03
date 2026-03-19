# pip install pytrends python-dotenv

import time
from dotenv import load_dotenv
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

load_dotenv()

def buscar_corona():
    pytrends = TrendReq(hl="pt-BR", tz=-180)

    termo = "Corona"

    try:
        pytrends.build_payload([termo], timeframe="today 12-m", geo="BR")
        related = pytrends.related_queries()

        bloco = related.get(termo, {})
        top = bloco.get("top")
        rising = bloco.get("rising")

        print("=" * 60)
        print(f"TERMOS ASSOCIADOS A: {termo}")
        print("=" * 60)

        print("\nTOP:")
        if top is not None and not top.empty:
            for _, row in top.head(10).iterrows():
                print(f"- {row['query']} ({row['value']})")
        else:
            print("Nenhum dado encontrado.")

        print("\nRISING:")
        if rising is not None and not rising.empty:
            for _, row in rising.head(10).iterrows():
                print(f"- {row['query']} ({row['value']})")
        else:
            print("Nenhum dado encontrado.")

    except TooManyRequestsError:
        print("Google Trends retornou 429 (muitas requisições). Tente novamente mais tarde.")
    except Exception as e:
        print(f"Erro ao consultar Google Trends: {e}")

if __name__ == "__main__":
    buscar_corona()