from pydantic import BaseModel, Field
from typing import Optional

class Transaction(BaseModel):
    step: int = Field(..., ge=0, description="Step represents a unit of time (1 step is 1 hour)")
    type: str = Field(..., description="Type of transaction (e.g. PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)")
    amount: float = Field(..., ge=0, description="Amount of the transaction in local currency")
    nameOrig: str = Field(..., description="Customer who initiated the transaction")
    oldbalanceOrg: float = Field(..., ge=0, description="Initial balance before the transaction")
    newbalanceOrig: float = Field(..., ge=0, description="New balance after the transaction")
    nameDest: str = Field(..., description="Customer who is the recipient of the transaction")
    oldbalanceDest: float = Field(..., ge=0, description="Initial balance recipient before the transaction")
    newbalanceDest: float = Field(..., ge=0, description="New balance recipient after the transaction")
    isFraud: int = Field(0, ge=0, le=1, description="Identifies fraud transaction (0 = Normal, 1 = Fraud)")
    isFlaggedFraud: int = Field(0, ge=0, le=1, description="Identifies illegal attempt to transfer massive amounts")
    producer_timestamp_ms: Optional[int] = Field(None, description="Producer ingestion timestamp in epoch milliseconds")
