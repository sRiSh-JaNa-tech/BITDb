import os
import zipfile
import tempfile
import gradio as gr

from pdf_extractor3 import func_extract


# ==================================================
# GLOBAL STORAGE FOR CURRENT BATCH
# ==================================================

generated_files = {}


# ==================================================
# EXTRACT MULTIPLE PDFs
# ==================================================

def extract_multiple(pdf_files):

    global generated_files

    generated_files = {}

    if not pdf_files:
        return (
            "Please select at least one PDF.",
            gr.update(choices=[]),
            "",
            None,
            None
        )

    output_files = []

    for pdf_file in pdf_files:

        try:

            # Extract PDF
            txt_path = func_extract(pdf_file)

            filename = os.path.basename(txt_path)

            # Store mapping
            generated_files[filename] = txt_path

            output_files.append(txt_path)

        except Exception as error:

            print(
                f"Error processing "
                f"{os.path.basename(pdf_file)}: {error}"
            )

    if not output_files:

        return (
            "No files were successfully processed.",
            gr.update(choices=[]),
            "",
            None,
            None
        )

    # ==================================================
    # CREATE ZIP
    # ==================================================

    zip_path = os.path.join(
        tempfile.gettempdir(),
        "extracted_texts.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        for txt_file in output_files:

            zip_file.write(
                txt_file,
                os.path.basename(txt_file)
            )

    filenames = list(generated_files.keys())

    status = (
        f"Successfully converted "
        f"{len(output_files)} PDF(s) to text."
    )

    return (
        status,
        gr.update(
            choices=filenames,
            value=filenames[0]
        ),
        "",
        None,
        zip_path
    )


# ==================================================
# PREVIEW SELECTED FILE
# ==================================================

def preview_file(filename):

    if not filename:
        return (
            "",
            None
        )

    txt_path = generated_files.get(filename)

    if not txt_path or not os.path.exists(txt_path):
        return (
            "File not found.",
            None
        )

    try:

        with open(
            txt_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        return (
            text,
            txt_path
        )

    except Exception as error:

        return (
            f"Unable to preview file: {error}",
            None
        )


# ==================================================
# UI
# ==================================================

with gr.Blocks(
    title="Fast PDF Extractor"
) as app:

    gr.Markdown(
        """
        # ⚡ Fast PDF Extractor

        Convert multiple PDF files into text quickly.
        Select a generated text file to preview and download it.
        """
    )

    # ==================================================
    # PDF INPUT
    # ==================================================

    pdf_input = gr.File(
        label="Select PDF files",
        file_types=[".pdf"],
        file_count="multiple",
        type="filepath"
    )

    extract_button = gr.Button(
        "⚡ Extract Text",
        variant="primary"
    )

    # ==================================================
    # STATUS
    # ==================================================

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    # ==================================================
    # GENERATED FILES
    # ==================================================

    gr.Markdown(
        "## 📄 Converted Text Files"
    )

    file_selector = gr.Dropdown(
        label="Select a text file",
        choices=[],
        interactive=True
    )

    # ==================================================
    # PREVIEW
    # ==================================================

    gr.Markdown(
        "## 👁️ File Preview"
    )

    text_preview = gr.Textbox(
        label="Text",
        lines=20,
        max_lines=30,
        interactive=False
    )

    # ==================================================
    # INDIVIDUAL DOWNLOAD
    # ==================================================

    individual_download = gr.File(
        label="Download selected text file"
    )

    # ==================================================
    # DOWNLOAD ALL
    # ==================================================

    gr.Markdown(
        "## 📦 Download All"
    )

    zip_download = gr.File(
        label="Download all text files (.zip)"
    )

    # ==================================================
    # EVENTS
    # ==================================================

    extract_button.click(
        fn=extract_multiple,
        inputs=pdf_input,
        outputs=[
            status,
            file_selector,
            text_preview,
            individual_download,
            zip_download
        ]
    )

    file_selector.change(
        fn=preview_file,
        inputs=file_selector,
        outputs=[
            text_preview,
            individual_download
        ]
    )


# ==================================================
# START APPLICATION
# ==================================================

app.launch()