# Video Series Presets

These YAML files are reusable topic and timeline configurations for synthetic long-form video generation. Each preset extends `base_long_memory.yaml` and can be passed directly to the pipeline.

## Presets

| Time span | Config | Theme |
| --- | --- | --- |
| 1 month | `one_month_lagos_startup_sprint.yaml` | Nigerian startup demo sprint in Lagos |
| 1 month | `one_month_mexico_city_supper_club.yaml` | Mexico City supper club preparation |
| 3 months | `three_month_marseille_relocation.yaml` | Moroccan French relocation and community adaptation |
| 3 months | `three_month_toronto_wedding_photography_season.yaml` | Indo-Canadian wedding photography season |
| 6 months | `six_month_shanghai_programmer_family_album.yaml` | East Asian programmer family/work/travel album |
| 6 months | `six_month_sao_paulo_community_theater.yaml` | Afro-Brazilian adult community theater rehearsal |
| 1 year | `one_year_london_accra_caregiving_career.yaml` | Ghanaian British caregiving and career year |
| 1 year | `one_year_new_orleans_music_collective.yaml` | Black Creole music collective over four seasons |

Run from `generation/`:

```bash
python -m video_generator.pipeline \
  --config configs/video_series_presets/one_month_lagos_startup_sprint.yaml \
  --metadata-only
```
