SportsVision is a hybrid deep learning system for human action recognition
in sports videos. It uses YOLOv8 for athlete detection, ResNet-50 for
spatial features, Bi-LSTM for temporal modeling, and self-attention for
frame weighting.

Anyone is welcome to contribute — bug fixes, documentation, model
improvements, or UI changes.

---

## Tech Stack

- Python 3.10+
- PyTorch 2.x + Torchvision
- Ultralytics YOLOv8
- OpenCV, NumPy, Pandas
- FastAPI (backend)
- Next.js / React (frontend)
- Git + GitHub

---

## Project Structure

SportsVision/
├── backend/          # FastAPI server + model pipeline
│   ├── main.py
│   ├── models/
│   └── requirements.txt
├── frontend/         # Next.js UI
│   ├── app/
│   ├── components/
│   └── package.json
└── README.md

---

## Local Setup

### Backend

    cd backend
    python -m venv venv
    venv\Scripts\activate        # Windows
    pip install -r requirements.txt
    cd ..
    python -m uvicorn backend.main:app --reload

Backend runs at http://127.0.0.1:8000

### Frontend

    cd frontend
    npm install
    npm run dev

Frontend runs at http://localhost:3000

Create a `.env.local` file inside `frontend/` with:

    NEXT_PUBLIC_API_URL=http://127.0.0.1:8000

---

## How to Contribute

1. Fork this repository
2. Clone your fork:
       git clone https://github.com/YOUR_USERNAME/SportsVision.git
3. Create a branch for your change:
       git checkout -b fix-something
4. Make your changes and test locally
5. Commit with a clear message:
       git commit -m "Fix: athlete ROI crop on edge frames"
6. Push your branch:
       git push origin fix-something
7. Open a Pull Request against `main`

---

## Commit Message Style

Keep messages short and clear. Some examples:

- `Add: video upload validation`
- `Fix: Bi-LSTM input shape mismatch`
- `Update: README setup steps`
- `Refactor: move model loading to separate file`

Avoid messages like "changes" or "update stuff".

---

## What We Need Help With

- Better athlete tracking when multiple players are in frame
- Data augmentation for low-light sports clips
- Frontend UI improvements (upload progress, error states)
- Documentation and code comments
- Testing on different sports categories

---

## Reporting Issues

If something is not working, open an issue and include:

- What you were trying to do
- Steps to reproduce
- What you expected
- What actually happened
- Screenshots or error logs if possible

---

## Code Guidelines

- Follow PEP 8 for Python
- Keep functions small and readable
- Add comments only where logic is not obvious
- Do not commit large files (videos, model weights) — use `.gitignore`
- Test your change before opening a PR

---

## Contact

Maintainer: Sankalp Bankar
Repo: https://github.com/SankalpBankar/SportsVision

For questions, open an issue or reach out to the maintainer directly.
