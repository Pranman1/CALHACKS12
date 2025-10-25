##
# Register Gym environments.
##

# Import isaaclab_tasks.utils only when needed to avoid import issues during module loading
def _register_environments():
    """Register environments when Isaac Sim is properly initialized."""
    try:
        from isaaclab_tasks.utils import import_packages
        
        # The blacklist is used to prevent importing configs from sub-packages
        _BLACKLIST_PKGS = []
        # Import all configs in this package
        import_packages(__name__, _BLACKLIST_PKGS)
    except ImportError:
        # Skip registration if Isaac Sim is not initialized
        pass

# Only register if Isaac Sim is available
try:
    import omni.log
    _register_environments()
except ImportError:
    # Isaac Sim not available, skip registration
    pass
