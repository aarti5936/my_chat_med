import csv
import asyncio
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
from model import qa_bot  # Import chatbot from model.py

# CSV Paths
QUESTIONS_CSV = "medical_qa.csv"
RESULTS_CSV = "results.csv"

# Load chatbot once (avoid reloading inside the loop)
chatbot = qa_bot()

# Load models once (avoid reloading inside the loop)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")  # Faster SBERT model

# ROUGE Score Calculation
def rouge_score(pred, gold):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(pred, gold)
    return scores["rougeL"].fmeasure

# **Optimized SBERT Similarity Calculation**
def semantic_similarity(pred, gold):
    embeddings = sentence_model.encode([pred, gold])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity  # Range: 0 (no match) to 1 (perfect match)

# **Faster BERTScore Calculation (Uses a Smaller Model)**
def compute_bertscore(pred, gold):
    P, R, F1 = bert_score([pred], [gold], lang="en", model_type="distilbert-base-uncased")  # Smaller model
    return F1.item()

# Get chatbot response
async def get_chatbot_answer(question):
    response = await chatbot.ainvoke(question)
    return response["result"]

# **Optimized Evaluation Function**
async def evaluate_chatbot():
    results = []

    try:
        with open(QUESTIONS_CSV, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            if "Question" not in reader.fieldnames or "Expected Answer" not in reader.fieldnames:
                raise KeyError("CSV file must contain 'Question' and 'Expected Answer' columns.")

            for row in reader:
                question = row["Question"].strip()
                expected_answer = row["Expected Answer"].strip()

                chatbot_answer = await get_chatbot_answer(question)

                # Compute evaluation metrics
                rouge = rouge_score(chatbot_answer, expected_answer)
                semantic_sim = semantic_similarity(chatbot_answer, expected_answer)  # SBERT Score
                bert = compute_bertscore(chatbot_answer, expected_answer)  # BERTScore

                results.append([question, expected_answer, chatbot_answer, bert, rouge, semantic_sim])

        # Save results
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Expected Answer", "Chatbot Answer", "BERT Score", "ROUGE Score", "Semantic Similarity"])
            writer.writerows(results)

        print(f"✅ Evaluation completed! Results saved to {RESULTS_CSV}")

    except KeyError as e:
        print(f"❌ CSV Error: {e}")
    except Exception as e:
        print(f"⚠️ An error occurred: {e}")

# Run the evaluation
if __name__ == "__main__":
    asyncio.run(evaluate_chatbot())
