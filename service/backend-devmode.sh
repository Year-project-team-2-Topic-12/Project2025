#!/bin/bash
alembic upgrade head
uvicorn main:app --reload