import torch
import torch.nn.functional as F

def encode_qual_data(data: torch.Tensor, noncont_dict: dict, source_col: int = None, grouped: bool = False):
    """
    Encode data using a single specification dict for all non-continuous variables.

    Args:
        data: (N, D) torch tensor
        noncont_dict: dict mapping column_index -> num_classes for ALL non-continuous columns
        source_col: optional int; if provided, identifies which column in noncont_dict is the source

    Returns:
        encoded_data: torch.Tensor ordered as [continuous | categorical | source]
        cat_cols: list of lists of ints for categorical one-hot groups
        source_cols: list of lists of ints for source one-hot groups (empty or single block)
    """
    if noncont_dict is None or len(noncont_dict) == 0:
        raise ValueError("noncont_dict must include at least one non-continuous column.")

    if data.dtype not in (torch.float32, torch.float64):
        data = data.to(torch.float64)

    num_rows, num_cols = data.shape
    noncont_dict = dict(noncont_dict)

    if source_col is not None and source_col not in noncont_dict:
        raise ValueError("source_col must be a key in noncont_dict if provided.")

    noncont_cols = sorted(noncont_dict.keys())
    continuous_cols = [c for c in range(num_cols) if c not in noncont_dict]
    categorical_cols = [c for c in noncont_cols if c != source_col]

    parts = []
    continuous_tensors = []
    for c in continuous_cols:
        continuous_tensors.append(data[:, c].unsqueeze(1))
    if len(continuous_tensors) > 0:
        parts.append(torch.cat(continuous_tensors, dim=1))

    def _ohe(col_tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
        values = torch.round(col_tensor).to(torch.long)
        min_val = int(values.min().item())
        shifted = values - min_val
        max_idx = int(shifted.max().item())
        if max_idx >= num_classes:
            raise ValueError(f"Detected class index {max_idx} but num_classes={num_classes} for one-hot encoding.")
        return F.one_hot(shifted, num_classes=num_classes).to(data.dtype)

    categorical_tensors = []
    for c in categorical_cols:
        categorical_tensors.append(_ohe(data[:, c], int(noncont_dict[c])))
    if len(categorical_tensors) > 0:
        parts.append(torch.cat(categorical_tensors, dim=1))

    source_tensors = []
    if source_col is not None:
        source_tensors.append(_ohe(data[:, source_col], int(noncont_dict[source_col])))
    if len(source_tensors) > 0:
        parts.append(torch.cat(source_tensors, dim=1))

    encoded_data = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    # Build blocks
    offset = 0
    if len(continuous_tensors) > 0:
        offset += len(continuous_cols)

    cat_cols_blocks = []
    for c in categorical_cols:
        width = int(noncont_dict[c])
        cat_cols_blocks.append(list(range(offset, offset + width)))
        offset += width

    source_cols_blocks = []
    if source_col is not None:
        width = int(noncont_dict[source_col])
        source_cols_blocks.append(list(range(offset, offset + width)))
        offset += width

    if grouped and len(cat_cols_blocks) > 0:
        start_idx = cat_cols_blocks[0][0]
        end_idx = cat_cols_blocks[-1][-1]
        cat_cols_blocks = [list(range(start_idx, end_idx + 1))]

    return encoded_data, cat_cols_blocks, source_cols_blocks


def learn_encodings(data: torch.Tensor, int_tol: float = 1e-6):
    """
    Infer a qualitative encoding specification (qual_dict) from the provided data.

    For each column that appears integer-like (within tolerance), compute the
    minimum and maximum rounded values and derive the number of classes as
    (max - min + 1). Returns a dict mapping column_index -> num_classes.

    Args:
        data: (N, D) torch tensor
        int_tol: tolerance for determining integer-like columns

    Returns:
        qual_dict: dict[int, int] mapping column index to inferred num_classes
    """
    if data.dtype not in (torch.float32, torch.float64):
        data = data.to(torch.float64)

    _, num_cols = data.shape
    qual_dict = {}

    for col_idx in range(num_cols):
        col = data[:, col_idx]
        is_integer_like = torch.all(torch.abs(col - torch.round(col)) < int_tol)
        if not is_integer_like:
            continue
        col_long = torch.round(col).to(torch.long)
        min_val = int(col_long.min().item())
        max_val = int(col_long.max().item())
        num_classes = max_val - min_val + 1
        if num_classes <= 0:
            continue
        qual_dict[col_idx] = num_classes

    return qual_dict


# def one_hot_encode_integer_columns(data: torch.Tensor, int_tol: float = 1e-6):
#     """
#     data: (N, D) torch tensor (float or int)
#     Returns:
#       encoded_data: torch.Tensor with integer columns one-hot encoded
#       column_info: list of dicts describing how each original column was treated
#     """
#     if data.dtype not in (torch.float32, torch.float64):
#         data = data.to(torch.float64)

#     num_rows, num_cols = data.shape
#     encoded_columns = []
#     column_info = []

#     for col_idx in range(num_cols):
#         col = data[:, col_idx]

#         is_integer_like = torch.all(torch.abs(col - torch.round(col)) < int_tol)
#         if is_integer_like:
#             col_long = torch.round(col).to(torch.long)
#             min_val = int(col_long.min().item())
#             max_val = int(col_long.max().item())
#             num_classes = max_val - min_val + 1

#             # Shift so the smallest value maps to 0, then one-hot
#             class_indices = col_long - min_val
#             ohe = F.one_hot(class_indices, num_classes=num_classes).to(data.dtype)
#             encoded_columns.append(ohe)

#             column_info.append({
#                 "column": col_idx,
#                 "type": "one_hot",
#                 "min_value": min_val,
#                 "max_value": max_val,
#                 "num_classes": num_classes
#             })
#         else:
#             encoded_columns.append(col.unsqueeze(1))
#             column_info.append({
#                 "column": col_idx,
#                 "type": "continuous"
#             })

#     encoded_data = torch.cat(encoded_columns, dim=1)
#     return encoded_data, column_info