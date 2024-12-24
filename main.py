import os
import re
import openai
import numpy as np
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer

#######################################
# Configuration
#######################################

OPENAI_MODEL = "gpt-3.5-turbo"
PLACEHOLDER_PATTERN = r"\{\{(.*?)\}\}"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# User-provided instructions (could be dynamic)
USER_INSTRUCTIONS = "Use a professional tone, summarize key points clearly, and maintain the formatting context."

#######################################
# Functions
#######################################


def load_template(template_path: str) -> Document:
    """Load the DOCX template."""
    return Document(template_path)


def extract_placeholders_from_template(doc: Document):
    """Extract placeholders from the DOCX template.

    Searches paragraph text and also table cells if needed.
    """
    placeholders = set()
    # Check paragraphs
    for p in doc.paragraphs:
        matches = re.findall(PLACEHOLDER_PATTERN, p.text)
        for m in matches:
            placeholders.add(m.strip())

    # Check tables (if any)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                matches = re.findall(PLACEHOLDER_PATTERN, cell.text)
                for m in matches:
                    placeholders.add(m.strip())

    return list(placeholders)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    text_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
    return "\n".join(text_content)


def load_input_documents(doc_dir: str):
    """Load reference documents (TXT/MD/PDF) from a directory."""
    texts = []
    for filename in os.listdir(doc_dir):
        path = os.path.join(doc_dir, filename)
        if filename.lower().endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                texts.append(text)
        elif filename.lower().endswith(".pdf"):
            pdf_text = extract_text_from_pdf(path)
            if pdf_text.strip():
                texts.append(pdf_text)
    return texts


def create_embedding_index(docs):
    """Create embeddings for the reference documents."""
    embeddings = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def get_relevant_docs(query, docs, embeddings, top_k=3):
    """Return top_k most relevant documents for the given query."""
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    scores = np.dot(embeddings, query_embedding.T).squeeze()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [docs[i] for i in top_indices]


def generate_llm_content(placeholder, instructions, references, max_retries=2):
    """Use the LLM to generate content for a given placeholder."""
    system_prompt = (
        "You are a helpful assistant that uses the provided references "
        "to produce well-structured, contextually relevant text matching the provided instructions."
    )

    user_prompt = f"""
Placeholder: {placeholder}
Instructions: {instructions}

Relevant References:
{chr(10).join(references)}

Please produce a coherent piece of text that fits into the template section.
If references are not sufficient, provide a best-effort summary or note that no relevant info was found.
"""
    for _ in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            content = response.choices[0].message.content.strip()
            if content:
                return content
        except Exception:
            pass
    return f"[Warning: No suitable content generated for {placeholder}]"


def replace_placeholder_in_paragraph(paragraph, placeholder, new_text):
    """Replace a placeholder in a single paragraph's text.
    For more complex formatting, you might need to rebuild runs rather than replacing paragraph.text.
    """
    old_text = paragraph.text
    new_paragraph_text = old_text.replace("{{" + placeholder + "}}", new_text)
    paragraph.text = new_paragraph_text


def replace_placeholders_in_doc(doc: Document, placeholder_to_text):
    """Replace placeholders in the entire DOCX, including tables."""
    # Replace in paragraphs
    for p in doc.paragraphs:
        for ph, val in placeholder_to_text.items():
            if f"{{{{{ph}}}}}" in p.text:
                replace_placeholder_in_paragraph(p, ph, val)

    # Replace in tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for ph, val in placeholder_to_text.items():
                    if f"{{{{{ph}}}}}" in cell.text:
                        cell.text = cell.text.replace("{{" + ph + "}}", val)

    return doc


#######################################
# Main Logic Example
#######################################


def main():
    template_path = "template.docx"  # The user-provided DOCX template
    docs_dir = "input_docs"  # Directory containing input documents

    # Load template
    template_doc = load_template(template_path)
    placeholders = extract_placeholders_from_template(template_doc)

    if not placeholders:
        print(
            "No placeholders found in the template. Please add placeholders like {{SectionName}}."
        )
        return

    # Load and process input docs (TXT, MD, PDF)
    docs = load_input_documents(docs_dir)
    if not docs:
        print("No input documents found.")
        return

    embeddings = create_embedding_index(docs)

    placeholder_to_text = {}
    for ph in placeholders:
        # Find relevant documents
        relevant_docs = get_relevant_docs(ph, docs, embeddings, top_k=3)
        # Generate content via LLM
        content = generate_llm_content(ph, USER_INSTRUCTIONS, relevant_docs)
        placeholder_to_text[ph] = content

    # Insert generated content into the template
    filled_doc = replace_placeholders_in_doc(template_doc, placeholder_to_text)

    # Save final output
    output_path = "final_output.docx"
    filled_doc.save(output_path)
    print(f"Document generated and saved to: {output_path}")


if __name__ == "__main__":
    main()
