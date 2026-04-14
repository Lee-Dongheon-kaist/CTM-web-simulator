# CTM Web Simulator

Cell Transmission Model (CTM) based traffic simulation web app built with Streamlit.

## Project Structure

- `app.py`: Streamlit web UI and 4-case comparison visualization
- `code.py`: CTM core simulation logic
- `ctm_params.py`: model parameters
- `ctm_inputs.py`: scenario inputs
- `Network_image.png`: network figure used in the UI

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually http://localhost:8501).

## Share on Local Network

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Others on the same network can access:

```text
http://YOUR_IPV4_ADDRESS:8501
```

## Deploy (Streamlit Community Cloud)

1. Push this project to GitHub.
2. Go to Streamlit Community Cloud and connect your GitHub repo.
3. Set entry point to `app.py`.
4. Deploy and share the generated public URL.

## GitHub Upload Quick Steps

```bash
git init
git add .
git commit -m "Initial commit: CTM web simulator"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Notes

- Make sure `Network_image.png` is included in the repository.
- If Windows firewall blocks access for network sharing, allow port 8501.
