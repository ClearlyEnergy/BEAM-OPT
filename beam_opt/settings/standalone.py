"""
Django settings for beam_opt project when running standalone

"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'kj4n1k2jnb9dn<208$232309nwsd923-$!@'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition

INSTALLED_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
]

ROOT_URLCONF = 'beam_opt.urls'

REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_SCHEMA_CLASS': 'rest_framework.schemas.coreapi.AutoSchema',
    'PAGE_SIZE': 25,
    'TEST_REQUEST_DEFAULT_FORMAT': 'json',
    'DATETIME_INPUT_FORMATS': (
        '%Y:%m:%d', 'iso-8601', '%Y-%m-%d'
    ),
}

# Tells BEAM-OPT that it is being run standalone. When running through BEAM this will be False
STANDALONE = True
