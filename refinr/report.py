"""
Génération des rapports de batch : un JSON détaillé par fichier + un
résumé HTML lisible pour l'ensemble du batch.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path

from jinja2 import Template

from .batch import BatchResult

_HTML_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport de traitement Refinr — {{ generated_at }}</title>
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 1.4rem; }
  .summary { margin-bottom: 1.5rem; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; background: white; }
  th, td { border: 1px solid #ddd; padding: 6px 10px; font-size: 0.85rem; text-align: left; vertical-align: top; }
  th { background: #f0f0f0; }
  .ok { color: #1a7f37; font-weight: 600; }
  .fail { color: #c0392b; font-weight: 600; }
  .warn { color: #a86500; }
  .steps { list-style: none; padding-left: 0; margin: 0; }
  .steps li { margin-bottom: 4px; }
  code { background: #eee; padding: 1px 4px; border-radius: 3px; }
</style>
</head>
<body>
  <h1>Rapport de traitement — {{ generated_at }}</h1>
  <div class="summary">
    <strong>{{ succeeded_count }}</strong> réussi(s), <strong>{{ failed_count }}</strong> échoué(s),
    profil de destination : <code>{{ profile_key }}</code>
  </div>
  <table>
    <thead>
      <tr>
        <th>Fichier</th>
        <th>Statut</th>
        <th>Tags analyse</th>
        <th>Gain staging</th>
        <th>LUFS entrée → sortie</th>
        <th>True peak sortie</th>
        <th>Chaîne appliquée</th>
        <th>Avertissements</th>
        <th>Durée</th>
      </tr>
    </thead>
    <tbody>
    {% for o in outcomes %}
      <tr>
        <td>{{ o.input_path }}</td>
        {% if o.success %}
        <td class="ok">OK</td>
        <td>{{ o.report.analysis_tags | join(', ') }}</td>
        <td>{{ "%.2f"|format(o.report.gain_staging_db) }} dB</td>
        <td>{{ "%.1f"|format(o.report.input_measurement.integrated_lufs) if o.report.input_measurement.integrated_lufs is not none else "n/a" }}
            → {{ "%.1f"|format(o.report.final_measurement.integrated_lufs) if o.report.final_measurement.integrated_lufs is not none else "n/a" }} LUFS</td>
        <td>{{ "%.2f"|format(o.report.final_measurement.true_peak_dbtp) }} dBTP</td>
        <td>
          <ul class="steps">
          {% for step in o.report.steps %}
            <li><strong>{{ step.role }}</strong>: {{ step.preset_name }}<br><small>{{ step.reason }}</small></li>
          {% endfor %}
          </ul>
        </td>
        <td class="warn">{{ o.report.warnings | join('<br>') | safe }}</td>
        <td>{{ o.report.duration_seconds }}s</td>
        {% else %}
        <td class="fail" colspan="7">ÉCHEC — voir JSON pour la trace complète</td>
        <td></td>
        {% endif %}
      </tr>
    {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""
)


def write_reports(result: BatchResult, output_dir: str | Path, profile_key: str) -> dict:
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    per_file_paths = []
    for outcome in result.outcomes:
        stem = Path(outcome.input_path).stem
        json_path = reports_dir / f"{stem}.report.json"
        payload = {
            "input_path": outcome.input_path,
            "success": outcome.success,
            "report": dataclasses.asdict(outcome.report) if outcome.report else None,
            "error": outcome.error,
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        per_file_paths.append(json_path)

    html_path = reports_dir / "summary.html"
    html = _HTML_TEMPLATE.render(
        generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        succeeded_count=len(result.succeeded),
        failed_count=len(result.failed),
        profile_key=profile_key,
        outcomes=result.outcomes,
    )
    html_path.write_text(html, encoding="utf-8")

    return {"html_summary": str(html_path), "per_file_json": [str(p) for p in per_file_paths]}
