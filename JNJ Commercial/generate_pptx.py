"""
generate_pptx.py
Creates an executive-ready, professional one-slider PPT presentation (.pptx)
illustrating the Johnson & Johnson Commercial Oncology Knowledge Graph (KG) Pipeline
and how the KG works to condition and ground next-question context.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_one_slider():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    # Use blank slide layout
    blank_slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_slide_layout)
    
    # ---------------------------------------------------------
    # Color Palette (Executive Dark Slate / J&J Precision Navy)
    # ---------------------------------------------------------
    BG_COLOR = RGBColor(11, 19, 36)         # Deep Navy (#0B1324)
    CARD_BG = RGBColor(19, 31, 56)          # Dark Slate Card (#131F38)
    CARD_BORDER = RGBColor(45, 66, 105)     # Muted Border (#2D4269)
    ACCENT_RED = RGBColor(225, 29, 72)      # J&J Red Accent (#E11D48)
    ACCENT_CYAN = RGBColor(14, 165, 233)    # Tech Cyan (#0EA5E9)
    ACCENT_GREEN = RGBColor(16, 185, 129)   # Emerald Compliant (#10B981)
    ACCENT_AMBER = RGBColor(245, 158, 11)   # Account Barrier Amber (#F59E0B)
    ACCENT_PURPLE = RGBColor(168, 85, 247)  # SLM Engine Purple (#A855F7)
    
    TEXT_WHITE = RGBColor(255, 255, 255)
    TEXT_LIGHT = RGBColor(226, 232, 240)    # Slate 200
    TEXT_MUTED = RGBColor(148, 163, 184)    # Slate 400
    TEXT_HIGHLIGHT = RGBColor(56, 189, 248) # Cyan highlight
    
    # ---------------------------------------------------------
    # Background Canvas
    # ---------------------------------------------------------
    bg_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5)
    )
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = BG_COLOR
    bg_shape.line.fill.background() # No border
    
    # ---------------------------------------------------------
    # Header Section
    # ---------------------------------------------------------
    # Brand Tag Badge
    tag_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(8.0), Inches(0.3))
    tf_tag = tag_box.text_frame
    tf_tag.word_wrap = True
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = "JOHNSON & JOHNSON INNOVATIVE MEDICINE • COMMERCIAL ONCOLOGY AI"
    p_tag.font.name = "Segoe UI"
    p_tag.font.size = Pt(9.5)
    p_tag.font.bold = True
    p_tag.font.color.rgb = ACCENT_RED
    
    # Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.62), Inches(8.8), Inches(0.55))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = "Knowledge-Graph Conditioned Next-Question Intelligence Pipeline"
    p_title.font.name = "Segoe UI"
    p_title.font.size = Pt(20)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE
    
    # Subtitle
    sub_box = slide.shapes.add_textbox(Inches(0.6), Inches(1.15), Inches(8.8), Inches(0.4))
    tf_sub = sub_box.text_frame
    tf_sub.word_wrap = True
    p_sub = tf_sub.paragraphs[0]
    p_sub.text = "How Neo4j Graph Ontology & Live Subgraphs Eliminate Hallucinations and Dynamically Ground the SLM in Call Context"
    p_sub.font.name = "Segoe UI"
    p_sub.font.size = Pt(11)
    p_sub.font.color.rgb = TEXT_MUTED
    
    # Top KPI Metrics Badge Container (Right Side)
    kpi_card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.6), Inches(0.4), Inches(3.15), Inches(1.1)
    )
    kpi_card.fill.solid()
    kpi_card.fill.fore_color.rgb = RGBColor(15, 23, 42)
    kpi_card.line.color.rgb = RGBColor(30, 41, 59)
    kpi_card.line.width = Pt(1)
    
    tf_kpi = kpi_card.text_frame
    tf_kpi.word_wrap = True
    tf_kpi.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    p_kpi1 = tf_kpi.paragraphs[0]
    p_kpi1.alignment = PP_ALIGN.CENTER
    run1 = p_kpi1.add_run()
    run1.text = "117 Nodes  •  352 Edges  •  8 Personas\n"
    run1.font.bold = True
    run1.font.size = Pt(10)
    run1.font.color.rgb = ACCENT_CYAN
    
    run2 = p_kpi1.add_run()
    run2.text = "100% Policy Fidelity  •  <300ms SLM TTFT\n"
    run2.font.bold = True
    run2.font.size = Pt(9.5)
    run2.font.color.rgb = ACCENT_GREEN
    
    run3 = p_kpi1.add_run()
    run3.text = "Zero Redundant Inquiries  •  Account Aware"
    run3.font.size = Pt(8.5)
    run3.font.color.rgb = TEXT_MUTED

    # ---------------------------------------------------------
    # 5 Pipeline Stage Cards (Left-to-Right Horizontal Flow)
    # ---------------------------------------------------------
    stages = [
        {
            "num": "STAGE 1",
            "title": "Rep Input & NLU Normalization",
            "accent": ACCENT_CYAN,
            "items": [
                ("Multi-Modal Utterance", "Voice transcription or chat text from field specialist (OS / FRM)."),
                ("Entity Normalization", "Resolves Doctor ('Dr. Anurag'), Brand ('INLEXZO'), Account ('Apollo Hospitals')."),
                ("Dialogue State Tracker", "Maintains multi-turn context, accumulated slots, and turn progression.")
            ],
            "footer": "Input: 'Met Dr Anurag at Apollo...'"
        },
        {
            "num": "STAGE 2",
            "title": "Deterministic Policy Gatekeeper",
            "accent": ACCENT_RED,
            "items": [
                ("PHI / PII Redaction", "100% privacy firewall strips patient identifiers before model exposure."),
                ("Toxicity Control", "Auto-normalizes 'toxicity' to 'safety concerns' per J&J compliance."),
                ("Hard Role Boundary", "Neo4j check: FRM blocked from clinical trial efficacy; OS blocked from pricing.")
            ],
            "footer": "Result: 100% Regulatory Guardrail"
        },
        {
            "num": "STAGE 3",
            "title": "Knowledge Graph Dynamic Context",
            "accent": ACCENT_AMBER,
            "items": [
                ("Dynamic Traversal", "Compares Covered Topics vs Role Roadmap -> targets next unaddressed duty."),
                ("Anti-Repetition Engine", "Guarantees AI never repeats questions already answered in history."),
                ("Account Barrier Intel", "Maps historical friction (Apollo = candidate flagging / prior auth delay).")
            ],
            "footer": "Dual-Case: Rep Ask vs Proactive Probe"
        },
        {
            "num": "STAGE 4",
            "title": "Live Entity-Rel Subgraph",
            "accent": ACCENT_GREEN,
            "items": [
                ("Turn Subgraph Synth", "Materializes active graph: Rep -> Account -> HCP -> Brand -> Barrier."),
                ("Relationship Mapping", "Validates connections between clinical status, access pathways & actions."),
                ("Audit & Visual Trace", "Generates Mermaid ER diagram for instant compliance explainability.")
            ],
            "footer": "Live Subgraph: 8 Typed Active Nodes"
        },
        {
            "num": "STAGE 5",
            "title": "Flat Prompt & SLM Generation",
            "accent": ACCENT_PURPLE,
            "items": [
                ("Flat Prompt Format", "Serializes Role, Brand, Contact, Barrier, Target Topic & Transcript."),
                ("Fine-Tuned SLM", "Merged Llama-3.2-1B served via vLLM with PagedAttention engine."),
                ("Next Question Output", "Emits single, clinically grounded, highly specific next question (<300ms).")
            ],
            "footer": "Output: 'What was the main discussion?'"
        }
    ]

    card_width = Inches(2.28)
    card_gap = Inches(0.18)
    start_left = Inches(0.6)
    card_top = Inches(1.72)
    card_height = Inches(3.95)

    for i, st in enumerate(stages):
        cur_left = start_left + i * (card_width + card_gap)
        
        # Outer Card Container
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, cur_left, card_top, card_width, card_height
        )
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)
        
        # Header accent bar inside card
        header_bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, cur_left + Inches(0.08), card_top + Inches(0.08), card_width - Inches(0.16), Inches(0.05)
        )
        header_bar.fill.solid()
        header_bar.fill.fore_color.rgb = st["accent"]
        header_bar.line.fill.background()
        
        # Card Text Content
        tb = slide.shapes.add_textbox(cur_left + Inches(0.1), card_top + Inches(0.18), card_width - Inches(0.2), card_height - Inches(0.25))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.05)
        tf.margin_right = Inches(0.05)
        tf.margin_top = Inches(0.05)
        tf.margin_bottom = Inches(0.05)
        
        # Stage Number
        p_num = tf.paragraphs[0]
        p_num.text = st["num"]
        p_num.font.name = "Segoe UI"
        p_num.font.size = Pt(8.5)
        p_num.font.bold = True
        p_num.font.color.rgb = st["accent"]
        p_num.space_after = Pt(2)
        
        # Stage Title
        p_t = tf.add_paragraph()
        p_t.text = st["title"]
        p_t.font.name = "Segoe UI"
        p_t.font.size = Pt(11)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_WHITE
        p_t.space_after = Pt(10)
        
        # Bullet Items
        for heading, desc in st["items"]:
            p_item = tf.add_paragraph()
            p_item.space_after = Pt(7)
            
            run_h = p_item.add_run()
            run_h.text = f"▸ {heading}\n"
            run_h.font.name = "Segoe UI"
            run_h.font.size = Pt(8.5)
            run_h.font.bold = True
            run_h.font.color.rgb = TEXT_HIGHLIGHT
            
            run_d = p_item.add_run()
            run_d.text = desc
            run_d.font.name = "Segoe UI"
            run_d.font.size = Pt(8)
            run_d.font.color.rgb = TEXT_LIGHT
            
        # Card Footer Badge
        p_foot = tf.add_paragraph()
        p_foot.space_before = Pt(8)
        run_f = p_foot.add_run()
        run_f.text = f"🏷️ {st['footer']}"
        run_f.font.name = "Segoe UI"
        run_f.font.size = Pt(7.5)
        run_f.font.italic = True
        run_f.font.color.rgb = st["accent"]

        # Connector Arrow between cards (except last card)
        if i < len(stages) - 1:
            arrow_left = cur_left + card_width + Inches(0.03)
            arrow_top = card_top + (card_height / 2) - Inches(0.12)
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW, arrow_left, arrow_top, Inches(0.12), Inches(0.24)
            )
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = RGBColor(94, 114, 155)
            arrow.line.fill.background()

    # ---------------------------------------------------------
    # Bottom Comparison / Architecture Deep-Dive Banner
    # ---------------------------------------------------------
    banner_top = Inches(5.82)
    banner_height = Inches(1.35)
    banner_width = Inches(12.133)
    
    banner = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), banner_top, banner_width, banner_height
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = RGBColor(15, 23, 42)
    banner.line.color.rgb = RGBColor(51, 65, 85)
    banner.line.width = Pt(1)

    # 3-Column Comparison Layout in Bottom Banner
    col_w = Inches(3.8)
    col_gap = Inches(0.25)
    
    # Box 1: Why Flat LLMs Fail
    tb_b1 = slide.shapes.add_textbox(Inches(0.8), banner_top + Inches(0.1), col_w, banner_height - Inches(0.2))
    tf_b1 = tb_b1.text_frame
    tf_b1.word_wrap = True
    p1 = tf_b1.paragraphs[0]
    p1.text = "⚠️ THE PROBLEM: NAIVE RAG / PURE LLMS"
    p1.font.bold = True
    p1.font.size = Pt(9.5)
    p1.font.color.rgb = RGBColor(239, 68, 68)
    
    p1_sub = tf_b1.add_paragraph()
    p1_sub.text = (
        "• Confuses role boundaries (OS asks about pricing; FRM discusses clinical trials).\n"
        "• Hallucinates questions for information already shared by the rep in Turn 1.\n"
        "• Oblivious to historical institutional friction at target hospital systems."
    )
    p1_sub.font.size = Pt(8.5)
    p1_sub.font.color.rgb = TEXT_LIGHT

    # Box 2: How Knowledge Graph Solves It
    tb_b2 = slide.shapes.add_textbox(Inches(0.8 + col_w + col_gap), banner_top + Inches(0.1), col_w, banner_height - Inches(0.2))
    tf_b2 = tb_b2.text_frame
    tf_b2.word_wrap = True
    p2 = tf_b2.paragraphs[0]
    p2.text = "🧠 THE SOLUTION: KNOWLEDGE GRAPH GROUNDING"
    p2.font.bold = True
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = ACCENT_CYAN
    
    p2_sub = tf_b2.add_paragraph()
    p2_sub.text = (
        "• Deterministic Neo4j Ontology: 44 duties, 13 exclusions, and compliance rules.\n"
        "• Dynamic Topic Traversal: Selects next unaddressed clinical duty sequentially.\n"
        "• Account Intelligence: Proactively injects account-specific friction inquiries."
    )
    p2_sub.font.size = Pt(8.5)
    p2_sub.font.color.rgb = TEXT_LIGHT

    # Box 3: Enterprise Business & Compliance Impact
    tb_b3 = slide.shapes.add_textbox(Inches(0.8 + 2 * (col_w + col_gap)), banner_top + Inches(0.1), col_w, banner_height - Inches(0.2))
    tf_b3 = tb_b3.text_frame
    tf_b3.word_wrap = True
    p3 = tf_b3.paragraphs[0]
    p3.text = "🚀 ENTERPRISE BUSINESS & AUDIT VALUE"
    p3.font.bold = True
    p3.font.size = Pt(9.5)
    p3.font.color.rgb = ACCENT_GREEN
    
    p3_sub = tf_b3.add_paragraph()
    p3_sub.text = (
        "• 100% Policy Verbatim Fidelity: Zero off-label or out-of-scope infractions.\n"
        "• Sub-300ms Inference TTFT: Seamless real-time voice interview speed.\n"
        "• Live Audit Subgraph: Dynamic Mermaid graph generated for every turn."
    )
    p3_sub.font.size = Pt(8.5)
    p3_sub.font.color.rgb = TEXT_LIGHT

    # Save Presentation
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "JNJ_Commercial_KG_Pipeline_One_Slider.pptx"
    )
    prs.save(output_path)
    print(f"[Success] PPTX presentation created at: {output_path}")
    return output_path

if __name__ == "__main__":
    create_one_slider()
