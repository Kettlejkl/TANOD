# app/api/auth/database.py
"""
Database operations for authentication
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
from datetime import datetime
import os

from .models import Base, User, UserSession


class AuthDatabase:
    """Authentication database manager"""
    
    def __init__(self, db_path='app/data/auth.db'):
        """Initialize database connection"""
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        print(f"✅ Auth database initialized: {db_path}")
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope for database operations"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    # ========================================================================
    # USER OPERATIONS
    # ========================================================================
    
    def create_user(self, username, email, password_hash, role='operator'):
        """Create a new user"""
        with self.session_scope() as session:
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                role=role
            )
            session.add(user)
            session.flush()
            
            return user.to_dict()
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        with self.session_scope() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                result = user.to_dict()
                result['password_hash'] = user.password_hash
                return result
            return None
    
    def get_user_by_username(self, username):
        """Get user by username"""
        with self.session_scope() as session:
            user = session.query(User).filter(User.username == username).first()
            if user:
                result = user.to_dict()
                result['password_hash'] = user.password_hash
                return result
            return None
    
    def get_user_by_email(self, email):
        """Get user by email"""
        with self.session_scope() as session:
            user = session.query(User).filter(User.email == email).first()
            if user:
                result = user.to_dict()
                result['password_hash'] = user.password_hash
                return result
            return None
    
    def get_all_users(self):
        """Get all users (without password hashes)"""
        with self.session_scope() as session:
            users = session.query(User).all()
            return [user.to_dict() for user in users]
    
    def update_last_login(self, user_id):
        """Update user's last login timestamp"""
        with self.session_scope() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.last_login = datetime.utcnow()
                return True
            return False
    
    def update_password(self, user_id, new_password_hash):
        """Update user password"""
        with self.session_scope() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.password_hash = new_password_hash
                return True
            return False
    
    def delete_user(self, user_id):
        """Delete a user and their sessions"""
        with self.session_scope() as session:
            # Delete user sessions first
            session.query(UserSession).filter(
                UserSession.user_id == user_id
            ).delete()
            
            # Delete user
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                session.delete(user)
                return True
            return False
    
    # ========================================================================
    # SESSION OPERATIONS
    # ========================================================================
    
    def create_session(self, user_id, token, expires_at):
        """Create a new session token"""
        with self.session_scope() as session:
            user_session = UserSession(
                user_id=user_id,
                token=token,
                expires_at=expires_at
            )
            session.add(user_session)
            session.flush()
            
            return user_session.to_dict()
    
    def get_session(self, token):
        """Get session by token"""
        with self.session_scope() as session:
            user_session = session.query(UserSession).filter(
                UserSession.token == token,
                UserSession.is_active == True
            ).first()
            
            if user_session:
                return user_session.to_dict()
            return None
    
    def verify_session_token(self, token):
        """Verify session token and return user if valid"""
        with self.session_scope() as session:
            user_session = session.query(UserSession).filter(
                UserSession.token == token,
                UserSession.is_active == True
            ).first()
            
            if not user_session:
                return None
            
            # Check if expired
            if user_session.expires_at < datetime.utcnow():
                user_session.is_active = False
                return None
            
            # Get user
            user = session.query(User).filter(
                User.id == user_session.user_id,
                User.is_active == True
            ).first()
            
            if user:
                result = user.to_dict()
                result['password_hash'] = user.password_hash
                return result
            
            return None
    
    def delete_session(self, token):
        """Delete/invalidate a session"""
        with self.session_scope() as session:
            user_session = session.query(UserSession).filter(
                UserSession.token == token
            ).first()
            
            if user_session:
                user_session.is_active = False
                return True
            return False
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        with self.session_scope() as session:
            deleted = session.query(UserSession).filter(
                UserSession.expires_at < datetime.utcnow()
            ).delete()
            
            return deleted


# Initialize global database instance
auth_db = AuthDatabase()
