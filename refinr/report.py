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

_HTML_TEMPLATE = Template("""
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
  .steps li { margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px dashed #ddd; }
  .steps li:last-child { border-bottom: none; }
  .step-measure { font-size: 0.78rem; color: #444; }
  .step-bands { list-style: none; font-size: 0.78rem; color: #555; margin: 4px 0 0 0; padding: 0; }
  .step-bands li { border: none; padding: 4px 0 4px 8px; margin: 0; border-left: 2px solid #ccc; }
  .band-summary { font-weight: 600; color: #333; }
  .band-reason { display: block; color: #666; margin-top: 1px; }
  details.diag { margin-bottom: 6px; }
  details.diag summary { cursor: pointer; font-size: 0.8rem; color: #333; }
  .diag-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 2px 12px; font-size: 0.76rem; color: #444; margin-top: 4px; }
  .diag-grid div { padding: 1px 0; }
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
        <td>
          <div><strong>{{ o.report.analysis_tags | join(', ') }}</strong></div>
          <details class="diag">
            <summary>Diagnostic complet</summary>
            <div class="diag-grid">
              <div>Centroïde spectral</div><div>{{ "%.0f"|format(o.report.diagnostic.spectral.spectral_centroid_hz) }} Hz</div>
              <div>Pente spectrale (tilt)</div><div>{{ "%+.2f"|format(o.report.diagnostic.spectral.tilt_db_per_octave) }} dB/oct</div>
              <div>Crest factor</div><div>{{ "%.1f"|format(o.report.diagnostic.dynamics.crest_factor_db) }} dB</div>
              <div>Écrêtage source</div><div>{{ "%.3f"|format(o.report.diagnostic.dynamics.clipping_ratio_pct) }} %</div>
              <div>Corrélation stéréo</div><div>{{ "%.2f"|format(o.report.diagnostic.dynamics.stereo_correlation) }}</div>
              <div>Loudness range</div><div>{{ "%.2f"|format(o.report.diagnostic.dynamics.loudness_range_lu) if o.report.diagnostic.dynamics.loudness_range_lu is not none else "n/a" }} LU</div>
              {% for band_name, energy_db in o.report.diagnostic.spectral.band_energy_db.items() %}
              <div>Énergie {{ band_name }}</div><div>{{ "%.1f"|format(energy_db) }} dB</div>
              {% endfor %}
            </div>
          </details>
        </td>
        <td>{{ "%.2f"|format(o.report.gain_staging_db) }} dB</td>
        <td>{{ "%.1f"|format(o.report.input_measurement.integrated_lufs) if o.report.input_measurement.integrated_lufs is not none else "n/a" }}
            → {{ "%.1f"|format(o.report.final_measurement.integrated_lufs) if o.report.final_measurement.integrated_lufs is not none else "n/a" }} LUFS</td>
        <td>{{ "%.2f"|format(o.report.final_measurement.true_peak_dbtp) }} dBTP</td>
        <td>
          <ul class="steps">
          {% for step in o.report.steps %}
            <li>
              <strong>{{ step.role }}</strong>: {{ step.preset_name }}<br>
              <small>{{ step.reason }}</small>
              {% if step.extra and step.extra.bands %}
              <ul class="step-bands">
              {% for band in step.extra.bands %}
                <li><span class="band-summary">{{ band.summary }}</span><span class="band-reason">{{ band.reason }}</span></li>
              {% endfor %}
              </ul>
              {% endif %}
              {% if step.pre_measurement and step.pre_measurement.integrated_lufs is not none %}
              <div class="step-measure">
                avant: {{ "%.1f"|format(step.pre_measurement.integrated_lufs) }} LUFS /
                {{ "%.2f"|format(step.pre_measurement.true_peak_dbtp) }} dBTP
                → après: {{ "%.1f"|format(step.post_measurement.integrated_lufs) if step.post_measurement.integrated_lufs is not none else "n/a" }} LUFS /
                {{ "%.2f"|format(step.post_measurement.true_peak_dbtp) if step.post_measurement.true_peak_dbtp is not none else "n/a" }} dBTP
              </div>
              {% endif %}
            </li>
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
""")


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
