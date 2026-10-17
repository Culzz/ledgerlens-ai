import type { Report } from "../api";
import { count, isFigure } from "../format";

// The labels match what the retrieval layer hands the model, so the highlight in
// the report and the citation in the reply are literally the same string.
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

export default function ReportReader({ report, cited, offered }: Props) {
  const [header, ...body] = report.table;

  // A column is set flush right when most of its cells hold figures, decided per
  // column rather than per cell so a column does not go ragged. The first column
  // is always the label column, the way a statement leads with its line items.
  const rightward =
    header?.map((_, column) => {
      if (column === 0) return false;
      const cells = body.map((row) => row[column] ?? "").filter((cell) => cell.trim() !== "");
      if (cells.length === 0) return false;
      return cells.filter(isFigure).length / cells.length >= 0.6;
    }) ?? [];

  return (
    <div className="pane-body">
      <div className="sheet">
        <div className="sheet-head">
          <span className="filename">{report.filename}</span>
          <span className="counts">
            {body.length} rows · {report.context.length} lines
          </span>
        </div>

        {header ? (
          <div className="table-scroll">
            <table className="statement">
              <thead>
                <tr>
                  {header.map((cell, column) => (
                    <th key={column} className={rightward[column] ? "right" : undefined}>
                      {cell || "\u00a0"}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {body.map((row, offset) => {
                  // Row 0 is the header, which is why the label starts at 1.
                  const label = tableLabel(offset + 1);
                  const isCited = cited.has(label);
                  const isOffered = !isCited && offered.has(label);
                  const state = isCited ? "cited" : isOffered ? "offered" : undefined;
                  return (
                    <tr key={offset} className={state}>
                      {row.map((cell, column) => (
                        <td key={column} className={rightward[column] ? "right" : undefined}>
                          {cell || "—"}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="hint">This report carries no table.</p>
        )}

        {report.context.length > 0 && (
          <>
            <span className="label prose-label">Text around the table</span>
            <ol className="prose">
              {report.context.map((line, index) => (
                <li key={index} className={cited.has(textLabel(index)) ? "cited" : undefined}>
                  <span className="idx">{index}</span>
                  <span>{line || "\u00a0"}</span>
                </li>
              ))}
            </ol>
          </>
        )}

        <p className="legend">
          {cited.size > 0 ? (
            <>
              <span className="swatch swatch--cited" /> cited by the answer
            </>
          ) : null}
          {offered.size > cited.size ? (
            <>
              <span className="swatch swatch--offered" /> retrieved, not used
            </>
          ) : null}
          {cited.size === 0 && offered.size === 0 ? `${count(body.length)} rows in the table.` : null}
        </p>
      </div>
    </div>
  );
}
