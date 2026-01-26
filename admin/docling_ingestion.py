import os
import re
import logging

from dotenv import load_dotenv
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions, AcceleratorDevice, AcceleratorOptions, PictureDescriptionApiOptions
from docling.datamodel.pipeline_options import EasyOcrOptions  

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_pdf_to_markdown(pdf_path: str, output_path: str) -> str:
  
  os.makedirs(output_path, exist_ok=True)

  model = "gpt-4-turbo"
  api_key = os.environ.get("OPENAI_API_KEY")
  picture_desc_option = PictureDescriptionApiOptions(
      url="https://api.openai.com/v1/chat/completions",
      headers={
          "Authorization": f"Bearer {api_key}",
          "Content-Type": "application/json",
      },
      params=dict(model=model, max_tokens=200, temperature=0.3),
      timeout=90,
      prompt="Describe this image in few sentences in a single paragraph"
  )

  pipeline_options = PdfPipelineOptions(
      do_ocr=True,
      do_table_structure=True,
      do_formula_enrichment=True,
      do_picture_description=True,
      picture_description_options=picture_desc_option,
      generate_picture_images=True,
      generate_page_images=True,
      images_scale=2,
      enable_remote_services=True,
      # table_structure_options={"do_cell_matching": True},
      ocr_options=EasyOcrOptions(lang=["en"]),
      accelerator_options=AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
  )

  format_options = {
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
  }

  converter = DocumentConverter(format_options=format_options)

  logger.info("Starting conversion..")
  result = converter.convert(str(pdf_path))

  markdown_text = result.document.export_to_markdown(image_mode="embedded")

  with open(output_path, "w", encoding="utf-8") as f:
    f.write(markdown_text)
  logger.info(f"Markdown saved to {output_path}")

  return markdown_text

def remove_base64_images(markdown_text: str):
    # pattern of base64 images
    pattern = r'!\[.*?\]\(data:image/.*?;base64,([A-Za-z0-9+/=\n]+)\)'
    cleaned_md = re.sub(pattern, "", markdown_text)
    return cleaned_md

if __name__ == "__main__":
    pdf_path = r"admin/researchpdfs/NIPS-2017-attention-is-all-you-need-Paper.pdf"
    output_file = r"admin/markdown_output/attention.md"

    markdown_text = convert_pdf_to_markdown(pdf_path=pdf_path, output_path=output_file)

    clean_markdown = remove_base64_images(markdown_text)
    clean_markdown_file = r"admin/markdown_output/attention_clean.md"

    with open(clean_markdown_file, "w", encoding="utf-8") as f:
        f.write(clean_markdown)
    logger.info(f"Clean markdown saved to {clean_markdown_file}")
      