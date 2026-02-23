import base64
import logging
from typing import Any, Dict

import requests
from haystack import component

logger = logging.getLogger(__name__)


@component
class DogFactFetcher:
    """Haystack component that retrieves random dog facts."""

    @component.output_types(fact=str)
    def run(self) -> Dict[str, Any]:
        try:
            response = requests.get("https://dogapi.dog/api/v2/facts", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and data["data"]:
                    fact = data["data"][0]["attributes"]["body"]
                    logger.info("Dog fact fetched successfully")
                    return {"fact": fact}
            return {"fact": "Не удалось получить факт о собаках в данный момент."}
        except Exception as exc:
            logger.error("Dog fact fetch failed: %s", exc)
            return {"fact": "Извините, не удалось получить факт о собаках."}


@component
class WeatherFetcher:
    """Haystack component that retrieves current weather by city."""

    @component.output_types(weather_info=str)
    def run(self, location: str) -> Dict[str, Any]:
        try:
            geocode_url = "https://nominatim.openstreetmap.org/search"
            geocode_params = {
                "q": location,
                "format": "json",
                "limit": 1,
                "accept-language": "ru",
            }
            headers = {"User-Agent": "TelegramWeatherBot/2.0"}

            geo_response = requests.get(
                geocode_url, params=geocode_params, headers=headers, timeout=10
            )
            if geo_response.status_code != 200 or not geo_response.json():
                return {"weather_info": f"Не удалось найти город: {location}"}

            geo_data = geo_response.json()[0]
            lat = geo_data["lat"]
            lon = geo_data["lon"]
            city_name = geo_data.get("display_name", location).split(",")[0]

            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m,wind_direction_10m"
                ),
                "timezone": "auto",
            }

            weather_response = requests.get(weather_url, params=weather_params, timeout=10)
            if weather_response.status_code != 200:
                return {
                    "weather_info": f"Не удалось получить информацию о погоде для {location}."
                }

            current = weather_response.json().get("current", {})
            weather_codes = {
                0: "Ясно",
                1: "Преимущественно ясно",
                2: "Переменная облачность",
                3: "Пасмурно",
                45: "Туман",
                48: "Изморозь",
                51: "Легкая морось",
                53: "Морось",
                55: "Сильная морось",
                61: "Небольшой дождь",
                63: "Дождь",
                65: "Сильный дождь",
                71: "Небольшой снег",
                73: "Снег",
                75: "Сильный снегопад",
                80: "Ливень",
                81: "Сильный ливень",
                82: "Очень сильный ливень",
                95: "Гроза",
                96: "Гроза с градом",
                99: "Сильная гроза с градом",
            }

            wind_dir = current.get("wind_direction_10m", "N/A")
            wind_dir_text = self._wind_direction(wind_dir)
            weather_desc = weather_codes.get(current.get("weather_code", 0), "Неизвестно")

            weather_info = (
                f"Погода в {city_name}:\n"
                f"🌡️ Температура: {current.get('temperature_2m', 'N/A')}°C "
                f"(ощущается как {current.get('apparent_temperature', 'N/A')}°C)\n"
                f"☁️ Состояние: {weather_desc}\n"
                f"💧 Влажность: {current.get('relative_humidity_2m', 'N/A')}%\n"
                f"💨 Ветер: {current.get('wind_speed_10m', 'N/A')} км/ч, "
                f"направление {wind_dir_text}"
            )
            return {"weather_info": weather_info}
        except requests.exceptions.Timeout:
            return {
                "weather_info": (
                    f"Превышено время ожидания при получении погоды для {location}. "
                    "Попробуйте позже."
                )
            }
        except Exception as exc:
            logger.error("Weather fetch failed: %s", exc)
            return {
                "weather_info": (
                    f"Извините, не удалось получить информацию о погоде для {location}."
                )
            }

    @staticmethod
    def _wind_direction(degrees: Any) -> str:
        if degrees is None or degrees == "N/A":
            return "N/A"
        directions = ["С", "СВ", "В", "ЮВ", "Ю", "ЮЗ", "З", "СЗ"]
        idx = int((float(degrees) + 22.5) / 45) % 8
        return directions[idx]


@component
class DogImageFetcher:
    """Haystack component that fetches and describes dog images."""

    def __init__(self, openai_api_key: str, openai_base_url: str):
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url

    @component.output_types(image_url=str, description=str)
    def run(self) -> Dict[str, Any]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get("https://dog.ceo/api/breeds/image/random", timeout=15)
                if response.status_code != 200:
                    if attempt < max_retries - 1:
                        continue
                    return {
                        "image_url": "",
                        "description": "Не удалось получить изображение собаки в данный момент.",
                    }

                data = response.json()
                if data.get("status") != "success":
                    if attempt < max_retries - 1:
                        continue
                    return {
                        "image_url": "",
                        "description": "Не удалось получить изображение собаки в данный момент.",
                    }

                image_url = data["message"]
                img_response = requests.get(image_url, timeout=30)
                if img_response.status_code != 200:
                    if attempt < max_retries - 1:
                        continue
                    return {
                        "image_url": image_url,
                        "description": "Найдено изображение, но не удалось загрузить его для описания.",
                    }

                image_base64 = base64.b64encode(img_response.content).decode("utf-8")
                description = self._describe_dog_image(image_base64)
                return {"image_url": image_url, "description": description}
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    continue
                return {
                    "image_url": "",
                    "description": (
                        "Извините, не удалось загрузить изображение собаки из-за таймаута. "
                        "Попробуйте еще раз."
                    ),
                }
            except Exception as exc:
                logger.error("Dog image fetch failed: %s", exc)
                if attempt < max_retries - 1:
                    continue
                return {
                    "image_url": "",
                    "description": "Извините, не удалось получить и описать изображение собаки.",
                }

        return {
            "image_url": "",
            "description": "Извините, не удалось получить изображение собаки после нескольких попыток.",
        }

    def _describe_dog_image(self, image_base64: str) -> str:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Проанализируй это изображение собаки и предоставь на русском языке: "
                                    "1) породу (или смесь), 2) краткую историю, 3) ключевые характеристики, "
                                    "4) интересные факты. Будь информативным, но кратким (3-4 предложения)."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Dog image description failed: %s", exc)
            return "Это похоже на собаку, но не удалось сгенерировать детальное описание."

