# Group 12 · CSE 543 Phishing URL Detection UI

This repo currently contains a front-end prototype for testing URLs against a phishing classifier.
It is designed to plug into a character-level CNN (or other model) once the backend is ready.

## Requirements

- Modern web browser (Chrome, Firefox, Edge, or Safari)
- Optional: Python 3 (only if you want to run a local static server)

## Run

Option 1: Open the file directly
- Open `index.html` in a browser

Option 2: Run a local static server (recommended)
```bash
cd /Users/ridhamshah/Documents/GitHub/Group-12-CSE-543
python -m http.server 5173
```
Then visit `http://localhost:5173`.

## Test It

Paste a URL into the input and click **Analyze**. If no backend is available, the UI uses a local
heuristic classifier and clearly labels it as mock.

Sample inputs:
- `https://example.com/login`
- `http://192.168.0.1/secure/login`
- `secure-payments-update.top/account/verify`
- `https://accounts.google.com`

## API Contract (Optional)

If you later expose a model service, the UI will call:

- `POST /api/classify`
- Request body:
  ```json
  { "url": "https://example.com/login" }
  ```
- Response body:
  ```json
  {
    "label": "Likely phishing",
    "score": 78,
    "confidence": 84,
    "signals": [
      { "title": "Punycode detected", "detail": "Potential homoglyph or IDN obfuscation." }
    ],
    "model": "CharCNN v0.1"
  }
  ```

If the API is unavailable, the UI automatically falls back to a local heuristic classifier (clearly labeled as mock).

## Note on Training Data

You are correct that synthetic/obfuscation-generated data is valuable for robustness testing. In practice,
phishing detectors typically train on real-world datasets and then add generated obfuscations to stress-test
and improve robustness, rather than training only on synthetic data.

## Files

- `index.html` — main UI
- `styles.css` — styling and layout
- `app.js` — UI behavior + mock classifier
