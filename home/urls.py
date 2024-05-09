from django.urls import path
from .views import video_feed, home, aasanPage, tracks,sittingAasanList,padmasana,lotus,recliningAasanList,standingAasanList,Virabhadrasana,Vrikshasana


urlpatterns = [
    path('', home, name='home'),
    path('tracks/aasanList/aasanPage/',aasanPage,name='aasanPage'),
    path('video_feed/', video_feed, name='video_feed'),
    path('tracks/',tracks,name='tracks'),
    path('tracks/sittingAasanList/',sittingAasanList,name='sittingAasanList'),
    path('tracks/sittingAasanList/padmasana/',padmasana,name='padmasana'),
    path('tracks/sittingAasanList/lotus/',lotus,name='lotus'),
    path('tracks/recliningAasanList/',recliningAasanList,name='recliningAasanList'),
    path('tracks/standingAasanList/',standingAasanList,name='standingAasanList'),
    path('tracks/standingAasanList/Virabhadrasana/',Virabhadrasana,name='Virabhadrasana'),
     path('tracks/standingAasanList/Vrikshasana/',Vrikshasana,name='Vrikshasana'),

    

    
    # path('collect_data/', collect_data, name='collect_data'),
    # path('train_model/', train_model, name='train_model'),
]
