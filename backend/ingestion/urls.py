from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("extract_tables/", views.extract_tables_api, name="extract_tables"),
    path("extract_text/", views.extract_text_api, name="extract_text"),
    path("upload_pdf/", views.upload_pdf_api, name="upload_pdf"),
    path("chunk_and_embed/", views.chunk_and_embed_api, name="chunk_and_embed"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

