## Предварительно
Понятное дело, сперва клонируем репозиторий.

Далее создаём чистое виртуальное украшение, а затем устанавливаем все зависимости через `poetry`

```
poetry install
```

А затем скачиваем данные через `dvc`

```
dvc pull
```

## Первое задание

Сперва запускаем скрипт

```
python code/hw_1/enhance_noise.py
```
Скрипт берёт на вход оригинальный файл со свипером `data/hw_1/sweeper_gt.wav`, свипер, записанный с колонок, `data/hw_1/sweeper_rec.wav` и файл с оригинальным белым шумом `data/hw_1/white_noise_gt.wav`.

В результате мы получим файл `output/hw_1/enh_white_noise.wav` - аудио с белым шумом, где мы откорректировали спектр в соответствии с АЧХ колонок.

Далее запускаем скрипт

```
python code/hw_1/restore_rec.py
```

Скрипт берёт на вход исходный улучшенный шум `output/hw_1/enh_white_noise.wav`, его запись с колонок `data/hw_1/white_noise_rec.wav` и запись с колонок тестового файла `data/hw_1/recording.wav`. Получает имульсный отклик из деконцолюции двух файлов с шумом, затем сворачиваем отклик с записью тестового файла и сохраняем `output/hw_1/restore.wav`.

Результаты, мягко говоря, не очень. Что можно улучшить:

1. Микрофон...
2. Тщательнее выровнять записи во времени, но попытки и так были сделаны.

## Второе задание

Запускаем скрипт

```
python code/hw_2.py
```

Код сгенерирует 4 файла с разным `SNR`: -5, 0, 5 и 10 -- и сохранит в папку
`/output/hw_2/`. Также прогонит их через доступные в `torchmetrics` метрики
`SignalDistortionRatio`, `ScaleInvariantSignalDistortionRatio` и
`PerceptualEvaluationSpeechQuality` и выведет результаты. Результаты работы
NISQA лежат в файле `/output/hw_2/NISQA_results.csv`, а результат работы `DNSMOS` представлены в виде
файла `/output/hw_2/dnsmos.csv`. Также все результаты представлены в таблице
ниже.




| файл | SNR | SDR | SI-SDR | PESQ | NISQA(mos_pred) | NISQA(noi_pred) | NISQA(dis_pred) | NISQA(col_pred) | NISQA(loud_pred) | DNSMOS | MOS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `snr-5.wav` | -5 | -5.0109906 | -5.026846 | 1.063826 | 0.68706834 | 1.3495014 | 2.8681333 | 1.3250866 | 1.4882853 | 2.305998 | 1 |
| `snr_0.wav` | 0 | -0.007477297 | -0.015080672 | 1.0710368 | 1.0275223 | 1.3248811 | 4.1299486 | 2.758282 | 2.3072891 | 2.40894 | 2 |
| `snr_5.wav` | 5 | 4.9965277 | 4.9915285 | 1.1423959 | 1.8978573 | 1.2495964 | 4.3652115 | 3.3259141 | 3.027142 | 2.67619 | 2 |
| `snr_10.wav` | 10 | 9.999419 | 9.995243 | 1.3214328 | 2.3220136 | 1.4812186 | 4.490019 | 3.8426468 | 3.3288708 | 3.061875 | 4 |
| `snr_-5_DeepFilterNet3.wav` | 5.3167653 | -0.3247673 | -1.7007704 | 1.1593353 | 3.0952358 | 3.0875075 | 4.0781465 | 3.6955316 | 3.8067248 | 3.363556 | 1 |
| `snr_0_DeepFilterNet3.wav` | 9.084919 | 8.933215 | 8.612325 | 1.7562816 | 3.9001713 | 3.557959 | 4.28046 | 4.0747895 | 4.150477 | 3.84388 | 2 |
| `snr_5_DeepFilterNet3.wav` | 12.760008 | 12.826108 | 12.638403 | 2.2123663 | 4.40212 | 4.252387 | 4.4855533 | 4.2658296 | 4.411184 | 3.96980 | 4 |
| `snr_10_DeepFilterNet3.wav` | 16.152878 | 16.38098 | 16.21658 | 2.6651316 | 4.507142 | 4.1689267 | 4.543676 | 4.2390294 | 4.4209867 | 4.09963 | 5 |


В нашем случае перцептуальные метрики меняются незначительно, аналитические же разнятся в широком спектре значений. На мой взгляд, будто аналитические метрики здесь показывают себя лучше. Так, DNSMOS оценивает все файлы одинаково, в том время как разница в качестве между худшим и лучшим файлами очеивдно и разительна.

Кажется, тут NISQA сильно занижает оценку качества.

## Третье задание

Запускаем скрипт

```
python code/hw_3.py
```

Код берёт на вход результат предыдущего задания из папки `output/hw_2`, поэтому их нужно запускать последовательно. В результате в папке `output/hw_3` появятся 4 файла -- улучшенные версии из предыдущего задания, т.е. результат работы `DeepFilterNet`. Также он прогоняет их через все достпнуые в `torchmetric` метрики и выводит результат. Результаты работы NISQA лежат в файле `/output/hw_3/NISQA_results.csv`, а результат работы `DNSMOS` представлены в виде файла `/output/hw_3/dnsmos.csv`.  Также все метрики сведены в таблице выше.

Перцептуальные метрики ведут себя достаточно забавным образом. Так, DNSMOS оценивает худший из результатов этого этапа лучше чем самый лучший файл из предыдущего задания, что объективно неправда. Да, шума меньше, но само качества звука стало хуже, "жёваным". Про NISQA можно сказать то же самое, хоть оно и оценивает достаточно адекватно файлы с высоким `MOS`. Аналитические же метрики более адекватно оценивают разницу между файлами.

Можно сделать вывод, что никакая метрика не является "панацеей", и порой более высокие значения любой из них могут ничего не говорить. Кажется, самый адекватный подход состоит в совокупном взгляде на все метрики, аналитические и перцептуальные, вместе.
