import os
from multiprocessing import freeze_support
from dotenv import load_dotenv

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered, save_output
from marker.config.parser import ConfigParser


def main():
    load_dotenv()

    input_filename = "hsbc.pdf"
    base_name = os.path.splitext(input_filename)[0]
    output_directory = "./hsbc_output"
    os.makedirs(output_directory, exist_ok=True)

    config = {
        "output_format": "markdown",
        "use_llm": True,
        "force_ocr": True,
        "llm_service": "marker.services.azure_openai.AzureOpenAIService",
        "azure_endpoint": "https://hkust.azure-api.net/openai",
        "azure_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "deployment_name": "gpt-4o-mini",
        "azure_api_version":os.getenv("AZURE_OPENAI_API_VERSION"),
    }

    config_parser = ConfigParser(config)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )

    rendered = converter(input_filename)
    save_output(rendered, output_directory, base_name)

    markdown_text, _, images = text_from_rendered(rendered)
    print(markdown_text[:2000])
    print(rendered.metadata)
    print(f"\n✅ Output successfully saved to: {output_directory}")


if __name__ == "__main__":
    freeze_support()
    main()