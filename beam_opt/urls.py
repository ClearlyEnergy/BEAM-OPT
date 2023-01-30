# !/usr/bin/env python
# encoding: utf-8

from beam_opt.views import optimize, preprocess, preprocess_and_optimize, version
from django.conf.urls import url, include

from rest_framework import routers

api_v3_router = routers.DefaultRouter()

app_name = 'beam_opt'

urlpatterns = [
    url(r'^', include(api_v3_router.urls)),
    url(r'^preprocess/$', preprocess, name='preprocess'),
    url(r'^optimize/$', optimize, name='optimize'),
    url(r'^preprocess_and_optimize/$', preprocess_and_optimize, name='preprocess_and_optimize'),
    url(r'^version/$', version, name='version'),
]
