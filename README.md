Tameer Milhem 322729443 , Luay Marzok 319111605
# RecruitFlow AI – Autonomous Recruitment Agent

RecruitFlow AI is an autonomous agent that streamlines candidate retrieval, evaluation, and ranking given a raw job listing.  
It combines job parsing, candidate retrieval from a vector DB, personalized questionnaires, synthetic answer simulation, and candidate scoring into one seamless pipeline.

---

## 🔹 How It Works

When you run:

```bash
python main.py
```

You’ll see the agent introduction:
![RecruitFlow AI Intro](https://drive.google.com/uc?export=view&id=1xXygVo6uksLb_jFFvCJE-7vmtXmSIkaO)

Simply paste in a job listing, and the full recruitment pipeline will execute automatically. At the end, results are saved to the data/runs/<timestamp>/ folder — both as a human-readable report.md and a structured final.json file.

---

The pipeline runs in fixed stages:
1.	Job Parsing – The raw job listing is transformed into structured JSON (skills, education, experience, and other attributes) using the GPT-4o API.
 
2.	Candidate Retrieval – The job is embedded and matched against pre-embedded resumes in Qdrant (vector DB).
 
3.	Questionnaire Generation – For each candidate, a tailored set of 4–5 questions is generated to clarify their skills and fit. The job listing and candidate information (parsed to minimize token usage) are provided to the GPT-4o API.
 
4.	Answer Simulation – Candidate answers are simulated using a separate LLaMA deployment. Since this step is token-intensive but doesn’t require the full power of GPT-4o, we use Llama-4-Maverick-17B-128E-Instruct-FP8, which provides strong results at a lower cost.
 
5.	Scoring – Each candidate receives scores (skills / experience / education / overall) with concise reasons using GPT 4o.
 
6.	Final Ranking – Results are reranked and written to data/runs/<timestamp>/report.md (human-readable) and final.json (structured).

All of these stages are orchestrated using LangChain, which ties together the different micro-agents (job parsing, retrieval, questionnaire generation, simulation, scoring, and reporting) into a single autonomous pipeline. This ensures each step passes structured outputs seamlessly into the next.

---

🔹 Data & Preprocessing

The candidate dataset was prepared and uploaded to Qdrant in advance. We started with scraped resumes in PNG format (sourced from Kaggle). These images were preprocessed and converted to text using Tesseract OCR, then parsed into structured data using the lightweight Meta-Llama-3-8B-Instruct model. The parsing was run on Kaggle with GPU support; due to GPU limits, we managed to parse ~700 resumes across various fields.

[Kaggle notebook link](https://www.kaggle.com/code/tameermilhem/pasing-resumes-llama)

To add more domain-specific candidates, we generated additional parsed resumes manually using top-tier LLM interfaces (outside of the API). We focused on seven fields: Data Science, Java Development, Supply Chain Analysis, Transportation Engineering, Chip Design, Full-Stack Development, and Machine Learning Engineering (≈40 resumes per field).

Prompt Used to generate the synthetic resumes :
```bash
System message:

You are a strict data generator. Output JSON ONLY (a single array of 9 objects). No prose, no markdown. Each object must have:

{

  "resume_id": "string_unique_slug",

  "resume_txt": "plain text resume content",

  "resume_json": {

    "doc_id": "same as resume_id",

    "source_path": "synthetic://<resume_id>.txt",

    "name": "First Last",

    "contact": {

      "email": "dummy email like firstname.lastname@example.com",

      "phone": "dummy phone like +1-555-123-4567",

      "location": "dummy city, country",

      "linkedin": "https://www.linkedin.com/in/<slug>",

      "github": "https://github.com/<slug>"

    },

    "skills": [],

    "experience": [

      {"title": null, "company": null, "start": null, "end": null, "bullets": []}

    ],

    "education": [

      {"degree": null, "field": null, "school": null, "year": null}

    ],

    "certifications": []

}

Rules:

* Domain focus: [DOMAIN]

* Generate exactly 9 resumes: 3 junior, 3 mid, 3 senior.

* Distribute responsibilities, years of experience, and technologies realistically across seniority levels.

* resume_txt: 400–700 words, with sections: Summary, Skills, Experience (bullets), Education, Projects/Certs.

* resume_json: must faithfully reflect resume_txt.

* All contact fields must contain fake but realistic data (never null).

* Use unique resume_id slugs like "de_junior_sql_etl_01".

* Output a single JSON array (9 objects). No trailing commas, no markdown fences.
```


Finally, we embedded the structured data (skills, experience, education) and uploaded it—together with the candidate JSONs and resume text payloads—into the Qdrant vector database.

[Embedding + upload notebook link](https://colab.research.google.com/drive/1Y5RPKtdNfP3YP8JmC1tc3vnui_l8UTXQ?usp=sharing)

Supporting materials are available in Google Drive:

[📂 Raw/Parsed resumes (PNGs → text → JSON)](https://drive.google.com/drive/folders/1OXh7VK88FQkMy67cFKRtJ3i7cIaCiRDA?usp=sharing)

[📂 Preprocessing code](https://www.dropbox.com/scl/fo/1xeahpup9gs3x0ry9jpu3/AJNYdEfq8cBYhDViDmBjIbk?rlkey=deohx7dqmqyivcrmy4uphjpjc&st=zlxb4eq2&dl=0)


Note: as mentioned, due to GPU limits on Kaggle, only a subset of resumes was parsed - the rest were synthetically produced.

---

🔹 Real-World Scenarios

RecruitFlow AI can be applied in two common company situations:

1.	New Job Posting
 
	•	The company posts a new job listing.
 
	•	The resumes are already parsed, embedded, and uploaded to the vector DB.
 
	•	RecruitFlow AI retrieves and ranks the most relevant candidates for that job.
 
 2.	Exploring Existing Talent Pool
 
	•	The company hasn’t posted a new job yet.
 
	•	But it already has a vector DB of past applicant resumes.
 
	•	RecruitFlow AI can check whether there are already suitable candidates available for upcoming openings.

---

🔹 Setup & Run

1.	Clone the repo:
 ```bash
 git clone https://github.com/tmeermilhem/recruitflow-ai
 cd recruitflow-ai
```
2.	Create virtual environment (Python 3.11):
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3.	Create .env file with your API key (the GPT-4o:
```bash
nano .env
```
Example:
```bash
API_KEY=API_KEY
```
Be sure to update your LLaMA API key in recruitflow-ai/agent/config.py (line 25).
Set the following value with your actual key:
```bash
LLAMA_API_KEY: str = "API_KEY"
```

4.	Run:
 ```bash
python main.py
```
Paste a job listing, end input with Ctrl-D (macOS/Linux) or Ctrl-Z then Enter (Windows).

🔹 Outputs

Results are saved in:
```bash
data/runs/<timestamp>/
   ├── report.md     # Human-readable summary
   └── final.json    # Structured candidate results
```

---
Token usage :
During Embedding phase (resumes) - a total of 248326 tokens were used.
When we were testing our agent :
- a total of 219,696 input tokens were used with GPT-4o.
- a total of 29,826 output tokens were used with GPT-4o.
- a total of 1,966 were used for the embedding model.
- there was an issue in our counting function - some tokens were not counted, but they only a fraction of the documented used tokens.

---

Job listings used for the examples : 

1)
We're hiring for a Supply Chain Analyst to join our team in Austin, Texas. In this role, you'll be responsible for analyzing our end-to-end supply chain processes to identify areas for improvement. You'll use data to optimize inventory levels, improve logistics efficiency, and enhance forecasting accuracy. You'll report directly to the Supply Chain Manager and play a key role in our continuous improvement initiatives.

Qualifications:

•	Education: Bachelor's degree in Supply Chain Management, Industrial Engineering, or a related field.
•	Experience: 3-5 years of experience in a supply chain, logistics, or operations role.
•	Technical Skills: Strong command of data analysis tools. Proficiency in SQL and at least one data visualization tool (e.g., Tableau, Power BI) is required.
•	Soft Skills: A knack for problem-solving, strong communication skills, and the ability to collaborate effectively with cross-functional teams.

2)
We're looking for a highly skilled Machine Learning Engineer to join our team in Austin, Texas. In this role, you'll be instrumental in bridging the gap between data science and software engineering by focusing on the operationalization of machine learning models. You'll be involved in the entire MLOps lifecycle, from building scalable data pipelines to deploying, monitoring, and maintaining models in production environments. We're seeking a team player who can take a model from a Jupyter Notebook to a robust, production-ready system.

Qualifications:

•	Experience: 3-5 years of professional experience in an MLOps, Machine Learning Engineering, or a similar role focused on model deployment.
•	Education: Bachelor's or Master's degree in Computer Science, Data Science, or a related quantitative field. A Ph.D. is a plus.
•	Technical Skills: Expert proficiency in Python. Hands-on experience with popular machine learning frameworks like TensorFlow or PyTorch. Strong knowledge of cloud platforms (AWS, GCP, or Azure), containerization (Docker), and orchestration (Kubernetes). Familiarity with CI/CD pipelines, data engineering, and MLOps tools (MLflow, Kubeflow).
•	Soft Skills: Strong problem-solving abilities, excellent communication skills for collaborating with data scientists and software engineers, and a passion for building scalable, reliable systems.

3)
We're seeking a passionate and skilled Full Stack Developer to join our team in Austin, Texas. In this role, you will be responsible for both front-end and back-end development, working on all stages of the software development lifecycle. You'll be building and maintaining a web application that serves a diverse and growing user base, so a strong grasp of modern technologies and a collaborative spirit are essential.

Qualifications:

•	Experience: 3-5 years of professional experience as a Full Stack Developer.
•	Education: Bachelor's degree in Computer Science, Software Engineering, or a related field.
•	Technical Skills: Proficiency in at least one modern front-end framework (e.g., React, Angular, Vue.js) and one back-end language/framework (e.g., Node.js, Python/Django, Ruby on Rails, Java/Spring). Strong knowledge of RESTful APIs, relational and NoSQL databases, and Git.
•	Soft Skills: Excellent problem-solving abilities, strong communication, and the capacity to work effectively in an Agile/Scrum team.











