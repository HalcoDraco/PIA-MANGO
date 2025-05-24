To install the required packages, run:
	pip install -r requirements.txt

To run the streamlit frontend, use:
	streamlit run frontend.py

To run the fastapi backend, use:
	fastapi run main_api.py

A file called ignored_api_key.txt located at 
the root of the repo must exist containing 
the API key for the replicate service.

The image generation models have been selected from 
the huggingface.co leaderboard found at:
	https://huggingface.co/spaces/ArtificialAnalysis/Text-to-Image-Leaderboard

This leaderboard is based on Quality ELO (a score based on
human evaluations via A-B testing), price and generation speed.
You can find a more detailed analysis here:
	https://artificialanalysis.ai/text-to-image