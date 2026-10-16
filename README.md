# MayaCardio Technical Assessment Submission
**Candidate:** Shreya Gaur  
**Profiles:** [GitHub](https://github.com/shreyagaur-10) | [LinkedIn](https://www.linkedin.com/in/shreyagaur1009) | [Portfolio](https://portfolio-dq2n.vercel.app/)  
**Title:** Where Is the Signal, and Where Is the Noise? (Photoplethysmography Signal Processing, Artifact Mitigation & Systems Reliability)

---

## Submission Directory Structure:
1. `MayaCardio_Signal_Processing_Assessment_Shreya_Gaur.pdf` — Primary formatted assessment report (converted directly from Microsoft Word).
2. `MayaCardio_Signal_Processing_Assessment_Shreya_Gaur.docx` — Complete editable Word assessment document with embedded publication-grade figures.
3. `MayaCardio_PPG_Signal_Processing_Assessment_Shreya_Gaur.ipynb` — Fully documented and reproducible Jupyter Notebook containing all data parsing, 4th-order zero-phase Butterworth filtering, and quantitative metrics.
4. `MayaCardio_Assessment_Comprehensive_Report.html` — Standalone, responsive interactive HTML report with embedded high-resolution figures.
5. `generate_figures.py` — Standalone Python script reproducing all signal processing, spectral analysis, and 300 DPI figures.
6. `MayaCardio_Signal_Processing_Case_Data_ppg.json` — Original supplied PPG data packets.
7. `figure1_all_five_streams.png` to `figure5_system_architecture_fault_tree.png` — Standalone 300 DPI publication plots.

---

## Candidate Background Highlights:
- **Project 1:** LocalDrop — Hyperlocal Marketplace Platform (Next.js, React, Node.js, Express, Supabase, JWT, Vercel)
- **Project 2:** ChurnRadar AI — End-to-End Customer Churn Prediction System (Python, Scikit-learn, XGBoost, LightGBM, SHAP, FastAPI, Streamlit)
- **Project 3:** Yatrimate AI — Multi-Agent Intelligent Travel Planning System (Python, LangGraph, LangChain, FastAPI, Groq, PostgreSQL, Tavily, Render)

---

## Execution & Reproducibility:
To re-run analysis and generate figures independently:
```bash
pip install numpy scipy matplotlib
python generate_figures.py
```
