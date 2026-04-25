"""
Gestionnaire des droits d'accès pour SecurAI Store.
Fait le lien entre les identités reconnues et les permissions de sécurité.
"""

class RightsManager:
    # Niveaux d'accès
    ADMIN = 'ADMIN'
    EMPLOYEE = 'EMPLOYEE'
    DENIED = 'DENIED'

    # Configuration des zones par défaut
    ZONES = ['entrance', 'stock', 'cashier', 'server']

    def __init__(self):
        """
        Initialise le gestionnaire avec les identités par défaut.
        """
        self.user_registry = {
            'Manager_Demo': self.ADMIN,
            'Employee_Demo': self.EMPLOYEE,
            'Unknown': self.DENIED
        }

        # Définition des permissions par niveau
        self.permissions_map = {
            self.ADMIN: {zone: True for zone in self.ZONES},
            self.EMPLOYEE: {
                'entrance': True,
                'stock': True,
                'cashier': False,
                'server': False
            },
            self.DENIED: {zone: False for zone in self.ZONES}
        }

        # Configuration UI par niveau
        self.ui_config_map = {
            self.ADMIN: {
                'color': '#00FF00',  # Vert éclatant
                'label': 'ADMINISTRATEUR - ACCÈS TOTAL',
                'icon': '🔓'
            },
            self.EMPLOYEE: {
                'color': '#FFFF00',  # Jaune
                'label': 'EMPLOYÉ - ACCÈS LIMITÉ',
                'icon': '🔑'
            },
            self.DENIED: {
                'color': '#FF0000',  # Rouge
                'label': 'ACCÈS REFUSÉ - INTRUS',
                'icon': '🚫'
            }
        }

    def add_identity(self, name, level):
        """
        Ajoute dynamiquement une identité au registre.
        """
        if level not in [self.ADMIN, self.EMPLOYEE, self.DENIED]:
            raise ValueError(f"Niveau d'accès invalide : {level}")
        self.user_registry[name] = level

    def get_access_level(self, name):
        """
        Retourne le niveau d'accès d'une identité.
        Retourne DENIED si l'identité est inconnue.
        """
        return self.user_registry.get(name, self.DENIED)

    def get_permissions(self, name):
        """
        Retourne le dictionnaire des permissions {zone: bool} pour une identité.
        """
        level = self.get_access_level(name)
        return self.permissions_map.get(level, self.permissions_map[self.DENIED])

    def is_authorized(self, name, zone):
        """
        Vérifie si une identité est autorisée dans une zone spécifique.
        """
        permissions = self.get_permissions(name)
        return permissions.get(zone, False)

    def get_ui_config(self, name):
        """
        Retourne la configuration UI associée au niveau d'accès de l'identité.
        """
        level = self.get_access_level(name)
        return self.ui_config_map.get(level, self.ui_config_map[self.DENIED])
