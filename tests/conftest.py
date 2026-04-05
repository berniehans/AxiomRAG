import pytest
import os

# Setar una variable de entorno de TESTING si fuera necesario para que Pydantic-Settings no busque el .env
os.environ["LLM_PROVIDER"] = "test"
