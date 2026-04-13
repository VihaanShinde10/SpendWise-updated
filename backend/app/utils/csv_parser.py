"""
CSV / Excel parser for bank statement uploads.
Supports flexible column name mapping.
"""
import pandas as pd
import io
from datetime import datetime
from typing import Optional
from loguru import logger


COLUMN_ALIASES = {
    "date": ["date", "txn date", "transaction date", "value date", "trans date",
             "booking date", "posting date", "transaction_date", "txn_date", "date_time"],
    "description": ["description", "narration", "particulars", "remarks", "details",
                    "transaction details", "trans details", "merchant", "raw_description",
                    "recipient_name", "recipient", "note", "transaction_remarks"],
    "amount": ["amount", "transaction amount", "debit/credit amount", "debit amount",
               "credit amount", "txn amount"],
    "debit": ["debit", "withdrawal", "dr", "debit amount"],
    "credit": ["credit", "deposit", "cr", "credit amount"],
    "balance": ["balance", "closing balance", "available balance", "running balance"],
    "method": ["method", "mode", "payment method", "type", "transaction type",
               "payment mode", "channel"],
}


def _find_column(df_cols: list, aliases: list) -> Optional[str]:
    """Find the first matching column from aliases (case-insensitive)."""
    lower_cols = {c.lower().strip(): c for c in df_cols}
    for alias in aliases:
        if alias.lower() in lower_cols:
            return lower_cols[alias.lower()]
    return None


def _detect_direction(row, debit_col: Optional[str], credit_col: Optional[str],
                      amount_col: Optional[str], desc: str) -> tuple:
    """Determine if a transaction is debit or credit and return amount."""
    amount = 0.0
    direction = "debit"

    if debit_col and credit_col:
        # Separate debit/credit columns
        debit_val = row.get(debit_col, None)
        credit_val = row.get(credit_col, None)
        try:
            debit_amt = float(str(debit_val).replace(',', '').strip()) if pd.notna(debit_val) and str(debit_val).strip() not in ('', '-', 'nan') else 0
        except (ValueError, TypeError):
            debit_amt = 0
        try:
            credit_amt = float(str(credit_val).replace(',', '').strip()) if pd.notna(credit_val) and str(credit_val).strip() not in ('', '-', 'nan') else 0
        except (ValueError, TypeError):
            credit_amt = 0

        if debit_amt > 0:
            amount = debit_amt
            direction = "debit"
        elif credit_amt > 0:
            amount = credit_amt
            direction = "credit"
    elif amount_col:
        raw = str(row.get(amount_col, 0)).replace(',', '').strip()
        try:
            amount = abs(float(raw))
        except (ValueError, TypeError):
            amount = 0
        # Infer direction from sign or description keywords
        try:
            if float(raw) < 0:
                direction = "debit"
            else:
                direction = "credit"
        except (ValueError, TypeError):
            desc_lower = desc.lower()
            direction = "credit" if any(w in desc_lower for w in ['salary', 'credit', 'received', 'refund']) else "debit"

    return amount, direction


def _detect_payment_method(desc: str, method_val: Optional[str]) -> str:
    if method_val and str(method_val).strip().upper() not in ('NAN', ''):
        mv = str(method_val).strip().upper()
        for m in ['UPI', 'IMPS', 'NEFT', 'ATM']:
            if m in mv:
                return m
    desc_upper = desc.upper()
    for m in ['UPI', 'IMPS', 'NEFT', 'ATM']:
        if m in desc_upper:
            return m
    return 'OTHER'


def parse_bank_statement(file_bytes: bytes, filename: str) -> list:
    """
    Parse a CSV or Excel bank statement and return a list of normalized transaction dicts.
    """
    try:
        if filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
        else:
            # Try multiple encodings
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, skip_blank_lines=True)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV file")
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop rows that are completely empty
    df = df.dropna(how='all')

    cols = df.columns.tolist()
    date_col = _find_column(cols, COLUMN_ALIASES["date"])
    desc_col = _find_column(cols, COLUMN_ALIASES["description"])
    amount_col = _find_column(cols, COLUMN_ALIASES["amount"])
    debit_col = _find_column(cols, COLUMN_ALIASES["debit"])
    credit_col = _find_column(cols, COLUMN_ALIASES["credit"])
    balance_col = _find_column(cols, COLUMN_ALIASES["balance"])
    method_col = _find_column(cols, COLUMN_ALIASES["method"])

    if not date_col or not desc_col:
        raise ValueError(
            f"Could not auto-detect required columns (date, description). "
            f"Found: {cols}. "
            f"Please ensure your CSV has columns like: date, description, amount."
        )

    if not amount_col and not (debit_col and credit_col):
        raise ValueError(
            "Could not find amount column(s). "
            "Need either 'amount' or separate 'debit'/'credit' columns."
        )

    transactions = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            # Parse date
            raw_date = row[date_col]
            if pd.isna(raw_date):
                skipped += 1
                continue
            if isinstance(raw_date, datetime):
                txn_date = raw_date
            else:
                raw_date_str = str(raw_date).strip()
                for fmt in [
                    '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y',
                    '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%d-%b-%Y',
                    '%d/%m/%y', '%m/%d/%y',
                ]:
                    try:
                        txn_date = datetime.strptime(raw_date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    try:
                        txn_date = pd.to_datetime(raw_date_str, dayfirst=True).to_pydatetime()
                    except Exception:
                        skipped += 1
                        continue

            # Build a rich description if multiple columns exist
            all_desc_parts = []
            main_desc = str(row[desc_col]).strip()
            if main_desc and main_desc.lower() != 'nan':
                all_desc_parts.append(main_desc)
            
            # Check for extra info in common columns like UPI_ID or Note
            for extra_col_alias in ["upi_id", "note", "remarks", "category_hint"]:
                found_col = _find_column(cols, [extra_col_alias])
                if found_col and found_col != desc_col:
                    val = str(row[found_col]).strip()
                    if val and val.lower() != 'nan' and val not in all_desc_parts:
                        all_desc_parts.append(val)
            
            desc = " | ".join(all_desc_parts)
            if not desc:
                skipped += 1
                continue

            amount, direction = _detect_direction(row, debit_col, credit_col, amount_col, desc)
            if amount <= 0:
                skipped += 1
                continue

            # Balance
            balance = None
            if balance_col:
                try:
                    bal_raw = str(row[balance_col]).replace(',', '').strip()
                    balance = float(bal_raw) if bal_raw not in ('nan', '', '-') else None
                except (ValueError, TypeError):
                    balance = None

            method = _detect_payment_method(desc, row.get(method_col) if method_col else None)

            transactions.append({
                "raw_description": desc,
                "amount": round(amount, 2),
                "direction": direction,
                "payment_method": method,
                "transaction_date": txn_date,
                "balance": balance,
            })
        except Exception as e:
            logger.warning(f"Skipped row: {e}")
            skipped += 1

    logger.info(f"Parsed {len(transactions)} transactions, skipped {skipped} rows from '{filename}'")
    return transactions
