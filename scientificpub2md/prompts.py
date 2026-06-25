"""The page-transcription prompt for the VLM.

This is the heart of the method: a single, carefully tuned instruction that makes the
vision-language model transcribe a rendered page *verbatim* while marking every section
heading as a ``## `` markdown header and dropping the noise (running heads, page numbers,
journal banners) and, by default, the back matter (references, acknowledgements, funding).
"""

# Base instruction shared by both back-matter modes.
_BASE = (
    "Extract the SCIENTIFIC CONTENT from this page of a research paper, in natural reading order, exactly as written.\n"
    "Mark EVERY section heading AND sub-heading as its own line beginning with '## ' (a markdown header), transcribing "
    "the heading text verbatim after the '## '. This includes top-level sections (Abstract, Introduction, Background, "
    "Materials and Methods, Methods, STAR Methods, Results, Discussion, Conclusions) AND their sub-headings "
    "(e.g. '## Bacterial Strains and Growth Conditions', '## Genome Sequencing', '## Statistical Analysis'). "
    "Only real headings get '## ' — never prefix body sentences, figure/table captions, or author/affiliation lines.\n"
)

# Always strip page furniture (running heads, page/line numbers, banners, DOIs, copyright).
_FURNITURE = (
    "OMIT entirely (do not transcribe): running headers/footers, page numbers, line numbers, journal/publisher banners "
    "(e.g. 'PLOS ONE', 'Nature'), DOIs, copyright/date lines"
)

# Default: also drop the back matter and skip all-references pages.
_BACKMATTER_DROP = (
    "; and the BACK MATTER — References, Acknowledgements, Funding, Author Contributions, Competing/Conflict of "
    "Interests, Data Availability, Supplementary file lists. If the page is ENTIRELY references or back-matter, output "
    "exactly 'SKIP'.\n"
)

# Opt-in: keep everything (still skip truly blank pages).
_BACKMATTER_KEEP = (
    ".\nTranscribe ALL remaining content including References and Acknowledgements. If the page is blank, output "
    "exactly 'SKIP'.\n"
)

_TAIL = (
    "For multi-column layouts read each column top-to-bottom. Keep figure/table captions as plain text. Do NOT summarize, "
    "paraphrase, translate, or add commentary.\n"
    "Output only the transcribed text, or 'SKIP'."
)


def page_prompt(keep_backmatter: bool = False) -> str:
    """Build the per-page transcription prompt.

    keep_backmatter=False (default) drops references/acknowledgements/funding etc. — the typical
    "clean full text" use case. Set True to transcribe the whole page including back matter.
    """
    return _BASE + _FURNITURE + (_BACKMATTER_KEEP if keep_backmatter else _BACKMATTER_DROP) + _TAIL
