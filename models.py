from sqlalchemy import Column, Integer, String, Float, DateTime
import datetime
from database import Base

class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    predicted_intent = Column(String)
    confidence = Column(Float)
    feedback = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
