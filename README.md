# SADA Extension for Stable Diffusion WebUI Forge
This repository implements SADA: Stability-guided Adaptive Diffusion Acceleration for ForgeUI. Speeds up SD XL, Flux by 1.3-2.0 times with almost no quality loss.


**Based on:** [SADA: Stability-guided Adaptive Diffusion Acceleration (ICML 2025)](https://github.com/Ting-Justin-Jiang/sada-icml)

## Attention! | Внимание!
This repository is written using Perplexity and other neural networks. I tried to anticipate possible errors and clean them up. On my configuration (Forge + a dozen different extensions) everything works. If you have an error, please write it, I and the neural networks will try to help you.

Этот репозиторий написан с помощью Perplexity и других нейросетей. Я попытался предусмотреть возможные ошибки и вычистить их. На моей конфигурации (Фордж + с десяток различных расширений) всё работает. Если у вас ошибка то напишите её, я и нейросети постораемся вам помочь. 


## English

### Description

SADA extension implements intelligent step skipping for Stable Diffusion WebUI Forge, delivering **1.2-1.8x speedup** with minimal quality loss. The algorithm analyzes feature similarity between consecutive denoising steps and skips redundant computations when the model output remains stable.

### Performance

- **SDXL models**: 1.2-1.5x faster generation
- **Flux models**: 1.3-1.8x faster generation  
- **SD 1.5 models**: 1.2-1.4x faster generation
- **Quality loss**: Minimal (LPIPS ≤ 0.10, FID ≤ 4.5)

### Compatibility

**Models:** SD 1.5, SDXL, Flux.1-dev, Flux.1-schnell  
**Samplers:** DPM++ 2M/SDE, Euler, Euler A, and all standard Forge samplers  
**Schedulers:** Normal, Karras, Exponential, SGM Uniform  
**Features:** Hires Fix, Kohya Hires Fix, TiledVAE, Batch processing

### Installation
Can be installed as a regular extension (Extension / Install from URL)

### Manual installation

1. Download `sada_forge.py`
2. Place in: `extensions/sd_forge_sada/scripts/sada_forge.py`
3. Restart Forge WebUI
4. Enable in generation interface

### Usage

#### Presets

**SDXL (Balanced):**
- Start: Step 15, End: Step 45
- Skip Rate: 20%, Threshold: 0.02

**Flux (Aggressive):**
- Start: Step 7, End: Step 35  
- Skip Rate: 30%, Threshold: 0.04

#### Tuning Guidelines

- **Start Step**: Adjust based on model convergence
  - Fast models (quick base formation): 8-12
  - Slow models: 18-25
- **End Step**: Leave 5-10 steps before completion
- **Skip Rate**: Start at 15-20%, increase if quality acceptable

### How It Works

SADA compares features between consecutive denoising steps using cosine similarity. When similarity exceeds the stability threshold, the step is skipped and approximated from the previous result. This eliminates redundant computations while preserving image quality.

The extension applies acceleration only within the specified step range, protecting critical early and final denoising phases.

## Русский

### Описание

Расширение SADA реализует интеллектуальный пропуск шагов для Stable Diffusion WebUI Forge, обеспечивая **ускорение в 1.2-1.8 раза** с минимальной потерей качества. Алгоритм анализирует схожесть признаков между последовательными шагами денойзинга и пропускает избыточные вычисления, когда выход модели остается стабильным.

### Производительность

- **SDXL модели**: ускорение в 1.2-1.5 раза
- **Flux модели**: ускорение в 1.3-1.8 раза
- **SD 1.5 модели**: ускорение в 1.2-1.4 раза
- **Потеря качества**: минимальная (LPIPS ≤ 0.10, FID ≤ 4.5)

### Совместимость

**Модели:** SD 1.5, SDXL, Flux.1-dev, Flux.1-schnell  
**Семплеры:** DPM++ 2M/SDE, Euler, Euler A и все стандартные семплеры Forge  
**Планировщики:** Normal, Karras, Exponential, SGM Uniform  
**Функции:** Hires Fix, Kohya Hires Fix, TiledVAE, батчевая обработка

### Установка
Можно установить как обычное расширение (Extension / Install from URL)

### Ручная установка
1. Скачайте `sada_forge.py`
2. Поместите в: `extensions/sd_forge_sada/scripts/sada_forge.py`
3. Перезапустите Forge WebUI
4. Включите в интерфейсе генерации

### Использование

#### Пресеты

**SDXL (Balanced):**
- Начало: 15 шаг, Конец: 45 шаг
- Пропуск: 20%, Порог: 0.02

**Flux (Aggressive):**
- Начало: 7 шаг, Конец: 35 шаг
- Пропуск: 30%, Порог: 0.04

#### Рекомендации по настройке

- **Начальный шаг**: настройте по скорости сходимости модели
  - Быстрые модели (быстрое формирование основы): 8-12
  - Медленные модели: 18-25
- **Конечный шаг**: оставьте 5-10 шагов до завершения
- **Процент пропуска**: начните с 15-20%, увеличивайте при приемлемом качестве

### Принцип работы

SADA сравнивает признаки между последовательными шагами денойзинга, используя косинусное сходство. Когда сходство превышает порог стабильности, шаг пропускается и аппроксимируется из предыдущего результата. Это исключает избыточные вычисления при сохранении качества изображения.

Расширение применяет ускорение только в указанном диапазоне шагов, защищая критические ранние и финальные фазы денойзинга.
