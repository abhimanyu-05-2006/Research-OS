"use client";


import { useRef, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";


const API =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";


type Citation = {
  citation_id: string;
  document_name: string;
  page_number: number;
  chunk_id: string;
  text: string;
};


type ResearchResponse = {
  answer: string;
  citations: Citation[];
  confidence: string;
  limitations: string[];
};


type UploadResponse = {
  status: string;
  document_hash: string;
  document_name?: string;
  chunks?: number;
  chunks_found?: number;
};


type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  documentName: string;
  citations?: Citation[];
};


type UploadedDocument = {
  name: string;
  hash: string;
  status: string;
  chunks?: number;
};


async function readResponse<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => null);


  if (!response.ok) {
    const detail = data?.detail;
    let message = `Request failed (${response.status}).`;


    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail
        .map(
          (item: { loc?: (string | number)[]; msg?: string }) => {
            const field = item.loc?.join(".") || "request";
            return `${field}: ${item.msg || "Invalid value"}`;
          },
        )
        .join("; ");
    }


    throw new Error(message);
  }


  if (!data) {
    throw new Error("Server returned an invalid response.");
  }


  return data as T;
}


export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const evidenceRef = useRef<HTMLElement>(null);


  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [selectedDocumentHash, setSelectedDocumentHash] =
    useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [sources, setSources] = useState<Citation[]>([]);
  const [evidenceDocumentName, setEvidenceDocumentName] =
    useState("");


  const busy = uploading || thinking;


  const selectedDocument = documents.find(
    (document) => document.hash === selectedDocumentHash,
  );


  function selectDocument(document: UploadedDocument) {
    if (busy) return;


    setSelectedDocumentHash(document.hash);
    setSources([]);
    setEvidenceDocumentName("");
    setError("");
    setNotice(`Selected PDF: ${document.name}`);
  }


  async function uploadDocument(
    event: ChangeEvent<HTMLInputElement>,
  ) {
    const file = event.target.files?.[0];
    event.target.value = "";


    if (!file || busy) return;


    setError("");
    setNotice("");


    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }


    if (file.size > 25 * 1024 * 1024) {
      setError("The PDF must be smaller than 25 MB.");
      return;
    }


    setUploading(true);


    try {
      const form = new FormData();
      form.append("file", file);


      const response = await fetch(
        `${API}/api/documents/upload`,
        {
          method: "POST",
          body: form,
        },
      );


      const data = await readResponse<UploadResponse>(response);


      if (
        typeof data.document_hash !== "string" ||
        !/^[a-f0-9]{64}$/.test(data.document_hash)
      ) {
        throw new Error(
          "The upload response did not include a valid document_hash.",
        );
      }


      const documentName = data.document_name || file.name;


      const uploadedDocument: UploadedDocument = {
        name: documentName,
        hash: data.document_hash,
        status: data.status,
        chunks: data.chunks ?? data.chunks_found,
      };


      setDocuments((previous) => {
        const alreadyListed = previous.some(
          (document) => document.hash === data.document_hash,
        );


        if (alreadyListed) {
          return previous.map((document) =>
            document.hash === data.document_hash
              ? uploadedDocument
              : document,
          );
        }


        return [...previous, uploadedDocument];
      });


      setSelectedDocumentHash(data.document_hash);
      setSources([]);
      setEvidenceDocumentName("");


      setNotice(
        data.status === "already_indexed"
          ? `This PDF was already indexed and is now selected: ${documentName}`
          : `PDF indexed and selected: ${documentName}`,
      );
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "The PDF could not be uploaded.",
      );
    } finally {
      setUploading(false);
    }
  }


  async function askQuestion(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();


    const text = question.trim();


    if (!text || busy) return;


    if (!selectedDocument) {
      setError("Please upload or select a PDF first.");
      return;
    }


    const documentHash = selectedDocument.hash;
    const documentName = selectedDocument.name;


    setError("");
    setNotice("");
    setQuestion("");
    setThinking(true);
    setSources([]);
    setEvidenceDocumentName("");


    setMessages((previous) => [
      ...previous,
      {
        id: crypto.randomUUID(),
        role: "user",
        text,
        documentName,
      },
    ]);


    try {
      const response = await fetch(
        `${API}/api/research/ask`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: text,
            top_k: 5,
            document_hash: documentHash,
          }),
        },
      );


      const data =
        await readResponse<ResearchResponse>(response);


      const answerSources = data.citations || [];


      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: data.answer,
          documentName,
          citations: answerSources,
        },
      ]);


      setSources(answerSources);
      setEvidenceDocumentName(documentName);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Research request failed.",
      );


      setQuestion(text);
    } finally {
      setThinking(false);
    }
  }


  function clearChat() {
    if (busy) return;


    setMessages([]);
    setSources([]);
    setEvidenceDocumentName("");
    setError("");
    setQuestion("");
    setNotice(
      "Chat cleared. Indexed PDFs and the current PDF selection are preserved.",
    );
  }


  function showSources(message: Message) {
    setSources(message.citations || []);
    setEvidenceDocumentName(message.documentName);


    evidenceRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }


  return (
    <main className="workspace">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-icon">R</span>
          <div>
            <h1>ResearchOS</h1>
            <p>Document intelligence</p>
          </div>
        </div>


        <button
          type="button"
          className="new-research"
          onClick={clearChat}
          disabled={busy}
        >
          + New research
        </button>


        <div className="section-label">
          DOCUMENTS <span>{documents.length}</span>
        </div>


        <div className="document-list">
          {documents.length === 0 ? (
            <p className="muted">
              No PDF has been uploaded in this session yet.
              Upload a PDF and select it.
            </p>
          ) : (
            documents.map((document) => {
              const isSelected =
                document.hash === selectedDocumentHash;


              return (
                <div
                  className="document-card"
                  key={document.hash}
                  style={
                    isSelected
                      ? {
                          outline:
                            "2px solid var(--accent, #2563eb)",
                          outlineOffset: "2px",
                        }
                      : undefined
                  }
                >
                  <span className="pdf-badge">PDF</span>


                  <div>
                    <p className="document-name">
                      {document.name}
                    </p>


                    <small>
                      {document.chunks !== undefined
                        ? `${document.chunks} chunks indexed`
                        : "Already indexed"}
                    </small>


                    <div style={{ marginTop: "8px" }}>
                      <button
                        type="button"
                        aria-pressed={isSelected}
                        onClick={() =>
                          selectDocument(document)
                        }
                        disabled={busy}
                        style={{
                          padding: "6px 10px",
                          borderRadius: "6px",
                          border: "1px solid #94a3b8",
                          background: isSelected
                            ? "#2563eb"
                            : "transparent",
                          color: isSelected
                            ? "#ffffff"
                            : "inherit",
                          cursor: busy
                            ? "not-allowed"
                            : "pointer",
                        }}
                      >
                        {isSelected
                          ? "✓ Selected"
                          : "Select PDF"}
                      </button>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>


        <div className="local-note">
          <span className="status-dot" />
          Local prototype
          <p>
            Questions are searched within the selected PDF.
            This is a shared local index; login and private
            workspaces are not enabled yet.
          </p>
        </div>
      </aside>


      <section className="research-panel">
        <header className="topbar">
          <div>
            <span className="eyebrow">
              RESEARCH WORKSPACE
            </span>
            <h2>Find answers. Follow the evidence.</h2>
            <p
              className="muted"
              style={{ overflowWrap: "anywhere" }}
            >
              {selectedDocument
                ? `Selected PDF: ${selectedDocument.name}`
                : "Upload or select a PDF first."}
            </p>
          </div>


          <button
            type="button"
            className="upload-button"
            onClick={() => inputRef.current?.click()}
            disabled={busy}
          >
            {uploading ? "Indexing PDF..." : "+ Upload PDF"}
          </button>


          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            onChange={uploadDocument}
            disabled={busy}
            hidden
            aria-label="Upload PDF"
          />
        </header>


        <div
          className="conversation"
          aria-live="polite"
        >
          {messages.length === 0 && (
            <div className="welcome">
              <div className="welcome-symbol">✦</div>


              <span className="eyebrow">
                YOUR DOCUMENTS, CONNECTED
              </span>


              <h2>Turn information into insight.</h2>


              <p>
                Upload a PDF, select it from the sidebar, and
                explore answers with sources retrieved from
                that PDF.
              </p>


              <div className="suggestions">
                {[
                  "Give a short summary of the selected PDF with page citations.",
                  "What are the important findings in the selected PDF?",
                  "Are there any contradictory claims in the selected PDF?",
                ].map((suggestion) => (
                  <button
                    type="button"
                    key={suggestion}
                    onClick={() =>
                      setQuestion(suggestion)
                    }
                    disabled={busy || !selectedDocument}
                  >
                    {suggestion} <span>↗</span>
                  </button>
                ))}
              </div>
            </div>
          )}


          {messages.map((message) => (
            <article
              key={message.id}
              className={`message ${message.role}`}
            >
              <div className="message-label">
                {message.role === "user"
                  ? "YOU"
                  : "RESEARCHOS"}
              </div>


              <p
                className="muted"
                style={{
                  fontSize: "12px",
                  overflowWrap: "anywhere",
                }}
              >
                PDF: {message.documentName}
              </p>


              <div className="message-body">
                {message.text}
              </div>


              {Boolean(message.citations?.length) && (
                <button
                  type="button"
                  className="source-link"
                  onClick={() => showSources(message)}
                >
                  View {message.citations?.length} retrieved
                  sources →
                </button>
              )}
            </article>
          ))}


          {thinking && (
            <div className="thinking">
              <span className="loading-dot" />
              Retrieving evidence from the selected PDF and
              generating an answer...
            </div>
          )}
        </div>


        <div className="composer-area">
          {error && (
            <div className="alert error" role="alert">
              {error}
            </div>
          )}


          {notice && (
            <div className="alert success">
              {notice}
            </div>
          )}


          <form
            className="composer"
            onSubmit={askQuestion}
          >
            <input
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              placeholder={
                selectedDocument
                  ? "Ask a question about the selected PDF..."
                  : "Upload or select a PDF first..."
              }
              minLength={3}
              maxLength={2000}
              required
              disabled={busy || !selectedDocument}
              aria-label="Research question"
            />


            <button
              type="submit"
              disabled={
                busy ||
                !selectedDocument ||
                question.trim().length < 3
              }
            >
              {thinking ? "Working..." : "Ask →"}
            </button>
          </form>


          <p className="composer-note">
            Questions are searched only within the selected
            PDF. Verify AI answers and citations against the
            original PDF. Claim-level verification is pending.
          </p>
        </div>
      </section>


      <aside
        ref={evidenceRef}
        className="evidence-panel"
      >
        <div className="evidence-heading">
          <span className="eyebrow">
            EVIDENCE EXPLORER
          </span>


          <h2>Retrieved sources</h2>


          <p style={{ overflowWrap: "anywhere" }}>
            {evidenceDocumentName
              ? `PDF used for the displayed answer: ${evidenceDocumentName}`
              : "Sources for an answer will appear here."}
          </p>
        </div>


        {sources.length === 0 ? (
          <div className="empty-evidence">
            <span>⌕</span>
            <p>
              No source excerpt is displayed yet. When a
              supported answer is generated, its sources will
              appear here. You can view sources for a previous
              answer using its “View sources” button.
            </p>
          </div>
        ) : (
          sources.map((source, index) => (
            <details
              className="source-card"
              key={`${source.citation_id}-${source.chunk_id}`}
              open={index === 0}
            >
              <summary>
                <span className="source-number">
                  {index + 1}
                </span>


                <span className="source-title">
                  {source.document_name}
                  <small>
                    Page {source.page_number}
                  </small>
                </span>
              </summary>


              <p>{source.text}</p>
            </details>
          ))
        )}


        <div className="evidence-footer">
          Excerpts are retrieved evidence, not independently
          verified proof.
        </div>
      </aside>
    </main>
  );
}