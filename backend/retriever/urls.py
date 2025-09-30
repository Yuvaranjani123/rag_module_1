from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('query/', views.query_document, name='query_document'),
    # Add more endpoints as needed
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)