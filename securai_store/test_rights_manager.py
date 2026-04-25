import pytest
from rights_manager import RightsManager

def test_manager_demo_has_admin_level():
    rm = RightsManager()
    assert rm.get_access_level('Manager_Demo') == RightsManager.ADMIN

def test_employee_demo_has_employee_level():
    rm = RightsManager()
    assert rm.get_access_level('Employee_Demo') == RightsManager.EMPLOYEE

def test_unknown_user_is_denied():
    rm = RightsManager()
    assert rm.get_access_level('Someone_Else') == RightsManager.DENIED

def test_admin_has_full_permissions():
    rm = RightsManager()
    permissions = rm.get_permissions('Manager_Demo')
    assert all(permissions.values()) is True

def test_employee_has_limited_permissions():
    rm = RightsManager()
    permissions = rm.get_permissions('Employee_Demo')
    assert permissions['entrance'] is True
    assert permissions['stock'] is True
    assert permissions['cashier'] is False
    assert permissions['server'] is False

def test_is_authorized_valid_zone():
    rm = RightsManager()
    assert rm.is_authorized('Manager_Demo', 'server') is True
    assert rm.is_authorized('Employee_Demo', 'server') is False

def test_add_identity_dynamically():
    rm = RightsManager()
    rm.add_identity('New_Boss', RightsManager.ADMIN)
    assert rm.get_access_level('New_Boss') == RightsManager.ADMIN
    assert rm.is_authorized('New_Boss', 'server') is True

def test_add_identity_invalid_level():
    rm = RightsManager()
    with pytest.raises(ValueError):
        rm.add_identity('Error_User', 'GOD_MODE')

def test_ui_config_admin():
    rm = RightsManager()
    config = rm.get_ui_config('Manager_Demo')
    assert config['color'] == '#00FF00'
    assert 'ADMIN' in config['label']
    assert config['icon'] == '🔓'

def test_ui_config_denied():
    rm = RightsManager()
    config = rm.get_ui_config('Unknown')
    assert config['color'] == '#FF0000'
    assert 'REFUSÉ' in config['label']
    assert config['icon'] == '🚫'
