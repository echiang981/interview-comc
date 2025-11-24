
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, TypedDict

from anthropic import APIStatusError, Anthropic


class TranslationConfig:
    source_language = "fr"
    target_language = "en"
    max_chunk_size = 2000  # Maximum characters per chunk
    overlap_paragraphs = 2  # Number of overlapping paragraphs between chunks
    # LLM settings
    temperature = 0.2
    top_p = 0.9
    max_output_tokens = 2048
    anthropic_model = "claude-3-5-sonnet-20240620"
    chunk_limit: Optional[int] = None

class GlossaryEntry(TypedDict):
    source: str
    behavior: Literal["keep_as_is", "translate_to"]
    target: str
    
Glossary = Dict[str, GlossaryEntry]

@dataclass
class Chunk:
    id: int
    text: str
    content_prefix: str
    start_paragraph_index: int
    end_paragraph_index: int

# Glossary management
def create_initial_glossary() -> Glossary:
    glossary: Glossary = {}
    
    # Add initial glossary entries if needed
    
    return glossary

# Paragraph splitting

def split_into_paragraphs(raw_text: str) -> list[str]:
    paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
    return paragraphs

# Turn paragraphs into list of chunks

def build_chunks(paragraphs: list[str], translationConfig: TranslationConfig) -> List[Chunk]:
    chunks: List[Chunk] = []

    start_index = 0
    chunk_id = 0
    curr_paragraphs = []
    curr_len = 0
    
    prev_chunk_end_index: Optional[int] = None
    
    for i, paragraph in enumerate(paragraphs):
        para_len = len(paragraph) + 2
        
        if curr_len + para_len > translationConfig.max_chunk_size and curr_paragraphs:
            chunk_text = '\n\n'.join(curr_paragraphs)
            # build content prefix if needed
            if prev_chunk_end_index and translationConfig.overlap_paragraphs > 0:
                overlap_start = max(prev_chunk_end_index - translationConfig.overlap_paragraphs, 0)
                overlap_paras = paragraphs[overlap_start:prev_chunk_end_index]
                content_prefix = '\n\n'.join(overlap_paras)
            else: 
                content_prefix = ""
            chunks.append(Chunk(
                id=chunk_id,
                text=chunk_text,
                content_prefix=content_prefix,
                start_paragraph_index=start_index,
                end_paragraph_index=i - 1
            ))
            chunk_id += 1
            
            # Prepare for next chunk with overlap
            prev_chunk_end_index = i
            start_index = i
            curr_paragraphs = [paragraph]
            curr_len = para_len
        else:
            curr_paragraphs.append(paragraph)
            curr_len += para_len
            
    # Add last chunk
    if curr_paragraphs:
        chunk_text = '\n\n'.join(curr_paragraphs)
        if prev_chunk_end_index and translationConfig.overlap_paragraphs > 0:
            overlap_start = max(prev_chunk_end_index - translationConfig.overlap_paragraphs, 0)
            overlap_paras = paragraphs[overlap_start:prev_chunk_end_index]
            content_prefix = '\n\n'.join(overlap_paras)
        else:
            content_prefix = ""
        chunks.append(Chunk(
            id=chunk_id,
            text=chunk_text,
            content_prefix=content_prefix,
            start_paragraph_index=start_index,
            end_paragraph_index=len(paragraphs) - 1
        ))
    
    return chunks

# Translate each chunk and update dynamic glossary

def make_translation_prompt(chunk: Chunk, glossary: Glossary, translationConfig: TranslationConfig) -> str:
    source_lang = getattr(translationConfig, "source_language", "source language")
    target_lang = getattr(translationConfig, "target_language", "target language")
    tone = getattr(translationConfig, "tone", None)
    register = getattr(translationConfig, "register", None)

    def _format_existing_glossary() -> str:
        if not glossary:
            return "None provided."
        lines: List[str] = []
        for entry in glossary.values():
            src = entry.get("source") or ""
            behavior = entry.get("behavior") or "translate_to"
            target = entry.get("target") or ""
            if behavior == "keep_as_is":
                lines.append(f"- {src} → keep as '{src}'.")
            else:
                lines.append(f"- {src} → translate as '{target}'.")
        return "\n".join(lines)

    chunk_text = getattr(chunk, "text", "")
    chunk_context = getattr(chunk, "content_prefix", "") or "None."
    chunk_id = getattr(chunk, "id", "unknown")
    start_idx = getattr(chunk, "start_paragraph_index", "unknown")
    end_idx = getattr(chunk, "end_paragraph_index", "unknown")

    instructions: List[str] = [
        f"Translate the provided text from {source_lang} to {target_lang}.",
        "Use the context to stay faithful to style, characters, and plot.",
        "Translate the `chunk_text` verbatim; do not skip sentences.",
        "Respect every rule in the existing glossary below.",
        "Identify any new proper names introduced in this chunk and "
        "suggest them as glossary entries for future consistency.",
        "Only return JSON in the format specified below. Do not add commentary or markdown fencing.",
        '{',
        '  "translation": "<chunk_text translated into the target language>",',
        '  "glossary": [',
        '    {"source": "<original term>", "behavior": "translate_to", "target": "<preferred translation>"},',
        '    {"source": "<term to keep identical>", "behavior": "keep_as_is", "target": ""}',
        '  ]',
        '}',
        "Use `translate_to` when a target rendering is required and `keep_as_is` when the source must remain unchanged.",
        "Only include NEW glossary entries; skip anything already provided.",
    ]
    if tone:
        instructions.insert(1, f"Maintain the author's {tone} tone.")
    if register:
        instructions.insert(2, f"Preserve the {register} register.")

    prompt_parts = [
        "You are a meticulous literary translator.",
        *instructions,
        "Existing glossary entries:",
        _format_existing_glossary(),
        "Context from preceding paragraphs (for reference only):",
        chunk_context,
        "Chunk metadata:",
        f"- chunk_id: {chunk_id}",
        f"- paragraph_range: {start_idx}–{end_idx}",
        "chunk_text:",
        chunk_text,
    ]

    return "\n".join(prompt_parts)

def call_translation_api(prompt: str, translationConfig: TranslationConfig) -> str:
    anthropic_key = getattr(translationConfig, "anthropic_api_key", None) or os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY for translation.")

    source_lang = getattr(translationConfig, "source_language", "source language")
    target_lang = getattr(translationConfig, "target_language", "target language")
    tone = getattr(translationConfig, "tone", None)
    style_guidelines = getattr(translationConfig, "style_guidelines", None)
    custom_system = getattr(translationConfig, "system_prompt", None)

    if custom_system:
        system_prompt = custom_system
    else:
        system_prompt = (
            "You are Claude, an expert literary translator. "
            f"Translate from {source_lang} to {target_lang} with perfect fidelity to meaning and tone."
        )
        if tone:
            system_prompt += f" Maintain the author's {tone} tone."
        if style_guidelines:
            system_prompt += f" Follow these additional instructions: {style_guidelines}."

    client = Anthropic(api_key=anthropic_key)
    temperature = getattr(translationConfig, "temperature", 0.2)
    top_p = getattr(translationConfig, "top_p", None)
    max_output_tokens = (
        getattr(translationConfig, "max_output_tokens", None)
        or getattr(translationConfig, "anthropic_max_output_tokens", None)
        or 2048
    )
    stop_sequences = getattr(translationConfig, "stop_sequences", None)
    model_name = getattr(translationConfig, "anthropic_model", None) or "claude-3-5-sonnet-20240620"

    request_params = {
        "system": system_prompt,
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if top_p is not None:
        request_params["top_p"] = top_p
    if stop_sequences:
        request_params["stop_sequences"] = stop_sequences

    try:
        response = client.messages.create(**request_params)
    except APIStatusError as exc:
        raise RuntimeError(f"Anthropic translation failed: {exc}") from exc

    text_blocks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
    translated_text = "".join(text_blocks).strip()
    if not translated_text:
        raise RuntimeError("Anthropic returned an empty response.")

    return translated_text


def translate_chunk(chunk: Chunk, glossary: Glossary, translationConfig: TranslationConfig) -> (str, Glossary):
    glossary = glossary or {}
    setattr(translationConfig, "glossary_entries", glossary)
    prompt = make_translation_prompt(chunk, glossary, translationConfig)
    llm_response = call_translation_api(prompt, translationConfig)

    try:
        parsed = json.loads(llm_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Translation output is not valid JSON: {exc}\nResponse: {llm_response}") from exc

    translation_text = parsed.get("translation")
    if not isinstance(translation_text, str) or not translation_text.strip():
        raise RuntimeError("Translation output missing 'translation' text.")
    translation_text = translation_text.strip()

    new_entries = parsed.get("glossary") or []
    updated_glossary: Glossary = dict(glossary)
    if isinstance(new_entries, list):
        for entry in new_entries:
            if not isinstance(entry, dict):
                continue
            source_term = entry.get("source")
            behavior = entry.get("behavior")
            target_term = entry.get("target", "")
            if (
                not source_term
                or behavior not in {"keep_as_is", "translate_to"}
                or source_term in updated_glossary
            ):
                continue
            updated_glossary[source_term] = {
                "source": source_term,
                "behavior": behavior,
                "target": target_term or "",
            }

    return translation_text, updated_glossary

# Recreate translated novel from translated chunks

def translate_book(raw_text, translationConfig) -> str:
    glossary = create_initial_glossary()
    paragraphs = split_into_paragraphs(raw_text)
    chunks = build_chunks(paragraphs, translationConfig)
    chunk_limit = getattr(translationConfig, "chunk_limit", None)
    
    translated_chunks = []
    for chunk in chunks:
        if chunk_limit is not None and len(translated_chunks) >= chunk_limit:
            break
        print(
            f"Translating chunk {chunk.id} "
            f"(paragraphs {chunk.start_paragraph_index}-{chunk.end_paragraph_index})..."
        )
        translated_chunk, glossary = translate_chunk(chunk, glossary, translationConfig)
        translated_chunks.append(translated_chunk)
        # Update glossary based on translated text if needed
        
    
    return "\n\n".join(translated_chunks)

def main():
    
    input_txt = "source/comc.txt"
    output_txt = "translated_output_en.txt"
    
    with open(input_txt, 'r', encoding='utf-8') as infile:
        raw_text = infile.read()
        
    translationConfig = TranslationConfig()
    translationConfig.source_language = "fr"
    translationConfig.target_language = "en"
    env_chunk_limit = os.getenv("TRANSLATION_CHUNK_LIMIT")
    if env_chunk_limit:
        try:
            translationConfig.chunk_limit = int(env_chunk_limit)
        except ValueError:
            raise ValueError("TRANSLATION_CHUNK_LIMIT must be an integer") from None
    else:
        translationConfig.chunk_limit = 1  # default to one chunk for quick testing
    
    print(
        f"Starting translation from {translationConfig.source_language} to {translationConfig.target_language} "
        f"using {input_txt}. Chunk limit: {translationConfig.chunk_limit or 'all'}."
    )
    translated = translate_book(raw_text, translationConfig)
    with open(output_txt, 'w', encoding='utf-8') as outfile:
        outfile.write(translated)
    print(f"Translation complete. Output written to {output_txt}.")


if __name__ == "__main__":
    main()
