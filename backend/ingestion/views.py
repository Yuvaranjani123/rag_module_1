from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
import os
import logging
from .utils import extract_and_save_tables, extract_text
from .service import ChunkerEmbedder
from dotenv import load_dotenv
load_dotenv()

from logs.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_pdf_api(request):
    pdf_file = request.FILES.get('pdf')
    if not pdf_file:
        logger.warning("No PDF file provided to upload_pdf_api")
        return Response({"error": "No PDF file provided"}, status=status.HTTP_400_BAD_REQUEST)
    save_path = os.path.join(settings.MEDIA_ROOT, pdf_file.name)
    with open(save_path, 'wb+') as f:
        for chunk in pdf_file.chunks():
            f.write(chunk)
    logger.info(f"PDF uploaded: {save_path}")
    return Response({"pdf_path": save_path, "pdf_name": pdf_file.name}, status=status.HTTP_200_OK)

@api_view(['POST'])
def extract_tables_api(request):
    pdf_path = request.data.get("pdf_path")
    output_dir = request.data.get("output_dir")

    if not pdf_path or not output_dir:
        logger.warning("extract_tables_api missing pdf_path or output_dir")
        return Response({"error": "pdf_path and output_dir required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        extract_and_save_tables(pdf_path, output_dir)
        logger.info(f"Tables extracted from {pdf_path} to {output_dir}")
        return Response({"message": "Tables extracted successfully"}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def extract_text_api(request):
    pdf_path = request.data.get("pdf_path")
    output_dir = request.data.get("output_dir")

    if not pdf_path or not output_dir:
        logger.warning("extract_text_api missing pdf_path or output_dir")
        return Response({"error": "pdf_path and output_dir required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        extract_text(pdf_path, output_dir)
        logger.info(f"Text extracted from {pdf_path} to {output_dir}")
        return Response({"message": "Text extracted successfully"}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def chunk_and_embed_api(request):
    """API endpoint to chunk and embed extracted content."""

    output_dir = request.data.get("output_dir")
    chroma_db_dir = request.data.get("chroma_db_dir")
    if not output_dir or not chroma_db_dir:
        logger.warning("chunk_and_embed_api missing output_dir or chroma_db_dir")
        return Response({"error": "output_dir and chroma_db_dir required"}, status=status.HTTP_400_BAD_REQUEST)
    try:
        AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
        AZURE_API_VERSION = os.getenv("AZURE_OPENAI_TEXT_VERSION", "2023-05-15")
        EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_TEXT_DEPLOYMENT_EMBEDDINGS")
        if not all([AZURE_ENDPOINT, AZURE_API_KEY, EMBEDDING_MODEL]):
            logger.error("Missing Azure OpenAI configuration in chunk_and_embed_api")
            return Response({"error": "Missing Azure OpenAI configuration"}, status=status.HTTP_400_BAD_REQUEST)
        chunker = ChunkerEmbedder(
            azure_endpoint=AZURE_ENDPOINT,
            azure_api_key=AZURE_API_KEY,
            azure_api_version=AZURE_API_VERSION,
            embedding_model=EMBEDDING_MODEL,
            chroma_persist_dir=chroma_db_dir,
            semantic_threshold=0.75
        )
        chunker.process_all_data(output_dir)
        logger.info(f"Chunking and embedding completed for {output_dir}, collection size: {chunker.collection.count()}")
        return Response({
            "message": "Chunking and embedding completed successfully",
            "collection_size": chunker.collection.count()
        }, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error in chunk_and_embed_api: {e}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
