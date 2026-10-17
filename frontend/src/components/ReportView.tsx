import type { Report } from "../api";

// The labels match what the retrieval layer hands the model, so the highlight
// and the citation in the reply are the same string.
export function tableLabel(index: number): string {
  return `table_${index}`;
}

export function textLabel(index: number): string {
  return `text_${index}`;
}

interface Props {
  report: Report;
  /** Units the answer cited, and units retrieval offered. */
  cited: Set<string>;
  offered: Set<string>;
}

export default function ReportView({ report, cited, offered }: Props) {
  const [header, ...body] = report.table;

  return (
    <div className="card">
      <h2>Report</h2>
      <div className="filename">{report.filename}</div>

      {header ? (
        <table className="report">
          <thead>
            <tr>
              {header.map((cell, column) => (
                <th key={column}>{cell || "\u00a0"}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {body.map((row, offset) => {
              // Row 0 is the header, which is why the label starts at 1 here.
              const index = offset + 1;
              const label = tableLabel(index);
              const classes = cited.has(label) ? "hit" : undefined;
              return (
                <tr key={index} className={classes} title={offered.has(label) ? "in the evidence" : undefined}>
                  {row.map((cell, column) => (
                    <td key={column}>{cell || "\u00a0"}</td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <p className="hint">This report has no table.</p>
      )}

      {report.context.length > 0 ? (
        <ul className="narrative">
          {report.context.map((line, index) => {
            const label = textLabel(index);
            return (
              <li key={index} className={cited.has(label) ? "hit" : undefined}>
                <span className="idx">{index}</span>
                <span>{line || "\u00a0"}</span>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="hint">This report has no narrative text.</p>
      )}
    </div>
  );
}
