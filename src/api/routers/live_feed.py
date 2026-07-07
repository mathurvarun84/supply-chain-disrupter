from typing import List
from fastapi import APIRouter
from src.api.schemas import NewsGroup, WeatherCity, LogLine, GanttRow
from src.api.fixtures import NEWS_GROUPS, WEATHER_CITIES, LOG_LINES, GANTT

router = APIRouter()


@router.get("/news", response_model=List[NewsGroup])
def get_news():
    return NEWS_GROUPS


@router.get("/weather", response_model=List[WeatherCity])
def get_weather():
    return WEATHER_CITIES


@router.get("/logs", response_model=List[LogLine])
def get_logs(run_id: str | None = None):
    return LOG_LINES


@router.get("/gantt", response_model=List[GanttRow])
def get_gantt(run_id: str | None = None):
    return GANTT
