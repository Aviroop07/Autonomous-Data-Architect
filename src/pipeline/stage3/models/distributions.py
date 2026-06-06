from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Any, Dict, Set, Tuple
import math
from datetime import datetime

class NumericRange(BaseModel):
    min: Optional[float] = Field(default=None, description="The minimum value for the range.")
    max: Optional[float] = Field(default=None, description="The maximum value for the range.")
    fact_references:Optional[List[int]] = Field(default_factory=list, description="IDs of the business facts supporting this range.")

    def _validate(self) -> List[str]:
        errors = []
        if self.min is not None and self.max is not None and self.min > self.max:
            errors.append(f"NumericRange min ({self.min}) cannot be greater than max ({self.max})")
        return errors

class DateRange(BaseModel):
    min: Optional[datetime] = Field(description="The starting date for the range.")
    max: Optional[datetime] = Field(description="The ending date for the range.")
    fact_references:Optional[List[int]] = Field(default_factory=list, description="IDs of the business facts supporting this range.")

    def _validate(self)->List[str]:
        errors = []
        if self.min is not None and self.max is not None and self.min>=self.max:
            errors.append(f"Min should be lower than Max, but currently: Min = {self.min} and Max = {self.max}, please provide fix.")
        return errors

class NormalDist(BaseModel):
    mean: float = Field(description="The arithmetic mean (average) of the distribution.")
    variance: float = Field(description="The measure of spread (sigma squared) of the distribution.")

    def _validate(self)->List[str]:
        errors = []
        if self.variance < 0:
            errors.append(f"Variance should be positive. It is {self.variance}. Please provide fix.")
        return errors

class LogNormalDist(BaseModel):
    mean: float = Field(description="The mean of the natural logarithm of the distribution.")
    variance: float = Field(description="The variance of the natural logarithm of the distribution.")

    def _validate(self)->List[str]:
        errors = []
        if self.variance < 0:
            errors.append(f"Variance should be positive. It is {self.variance}. Please provide fix.")
        return errors

class PoissonDist(BaseModel):
    lam: float = Field(description="The average rate (lambda) of occurance.")
    def _validate(self) -> List[str]:
        return []

class ZipfDist(BaseModel):
    a: float = Field(description="The skewness parameter for the Zipf distribution.")
    def _validate(self) -> List[str]:
        return []

class CategoricalDist(BaseModel):
    values: Set[Any] = Field(description="The set of all possible categorical labels.")
    weights: Dict[Any, float] = Field(description="Mapped probabilities for each value. Must sum to 1.0.")

    def _validate(self)->List[str]:
        errors = []
        if self.weights is not None:
            weight_keys = set(self.weights.keys())
            if not self.values.issubset(weight_keys) or not weight_keys.issubset(self.values):
                errors.append(f"The list of values and they keys in the `weights` paramter should be same, but they are not. Currently, values = {self.values} and keys in weight = {weight_keys}. Please provide fix.")
                return errors
            if sum(self.weights.values())!=1:
                errors.append(f"The weights given to the values should sum up to be 1, but they are not. Please provide fix.")
        return errors

class UnivariateDist(BaseModel):
    table_name: str = Field(description="The table that contain the column with this distribution.")
    column_name: str = Field(description="The column name.")
    distribution: Optional[Union[NumericRange, DateRange, NormalDist, LogNormalDist, PoissonDist, ZipfDist, CategoricalDist]] = Field(description="The structured distribution definition.")
    distribution_ref:Optional[List[int]] = Field(default_factory=list, description="IDs of the business facts supporting this distribution.")

    def _validate(self, schema: Any) -> List[str]:
        errors = []
        t_map = {t.name.upper(): t for t in schema.tables}
        if self.table_name.upper() not in t_map:
            errors.append(f"Table '{self.table_name}' not found in schema.")
            return errors

        table = t_map[self.table_name.upper()]
        col_map = {c.name.lower(): c for c in table.columns}
        col_obj = col_map.get(self.column_name.lower())

        if not col_obj:
            errors.append(f"Column '{self.column_name}' not found in table '{self.table_name}'.")
        else:
            # Check if PK
            if self.column_name.lower() == table.pk.lower():
                errors.append(f"Distribution assigned to Primary Key '{self.column_name}' in table '{self.table_name}'. PK columns cannot have statistical distributions.")

            # [STRICT TYPE CHECK]
            raw_d_type = col_obj.data_type.upper() if col_obj.data_type else "VARCHAR"
            d_type = raw_d_type.split("(")[0].strip()
            numeric_types = {"INT", "INTEGER", "FLOAT", "DECIMAL", "NUMERIC", "DOUBLE", "REAL", "BIGINT", "SMALLINT", "TINYINT"}
            date_types = {"DATE", "DATETIME", "TIMESTAMP"}

            if isinstance(self.distribution, (NumericRange, NormalDist, LogNormalDist, PoissonDist, ZipfDist)):
                if d_type not in numeric_types:
                    errors.append(f"Type mismatch: Distribution '{type(self.distribution).__name__}' requires a numeric column, but '{self.column_name}' is {d_type}.")
            elif isinstance(self.distribution, DateRange):
                if d_type not in date_types:
                    errors.append(f"Type mismatch: DateRange requires a DATE or TIMESTAMP column, but '{self.column_name}' is {d_type}.")

        # Validate the inner distribution if it exists
        if self.distribution and hasattr(self.distribution, "_validate"):
             errors.extend(self.distribution._validate())

        return errors
