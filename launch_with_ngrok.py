import streamlit.web.cli as stcli
import sys
from pyngrok import ngrok
import os

def main():
    # Configuration ngrok
    ngrok.set_auth_token("")  # Remplacez par votre token
    
    # Démarrer le tunnel ngrok
    public_url = ngrok.connect(8501)
    print("\n" + "="*70)
    print("APPLICATION STREAMLIT ACCESSIBLE VIA NGROK")
    print("="*70)
    print(f"\nURL PUBLIQUE : {public_url}")
    print(f"\nPartagez cette URL avec vos supérieurs")
    print("\nAPPUYEZ SUR CTRL+C POUR ARRETER\n")
    print("="*70 + "\n")
    
    # Lancer Streamlit
    sys.argv = ["streamlit", "run", "dashboard_forecasting.py", 
                "--server.port", "8501"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()