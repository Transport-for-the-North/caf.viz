{{ objname | escape | underline }}

{% if objname in show_inherited|default([]) -%}
   {%- set include_inherited_methods = True -%}
   {%- set include_inherited_attributes = True -%}
..
   This class documentation includes inherited methods and attributes
   because it is listed in the 'show_inherited' autosummary_context variable.
   show_inherited = {{ show_inherited }}.
{% elif objname in exclude_inherited|default([]) -%}
   {%- set include_inherited_methods = False -%}
   {%- set include_inherited_attributes = False -%}
..
   This class documentation excludes inherited methods and attributes
   because it is listed in the 'exclude_inherited' autosummary_context variable.
   exclude_inherited = {{ exclude_inherited }}.
{%- endif %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {%- if include_inherited_attributes|default(false) or item not in inherited_members %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
   {% for item in methods %}
      {%- if include_inherited_methods|default(false) or item not in inherited_members %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}


   {% block attribute_docs %}
   {% if attributes %}
   .. rubric:: Attributes Documentation
   {% for item in attributes %}
   {%- if include_inherited_attributes|default(false) or item not in inherited_members %}
   .. autoattribute:: {{ name }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. _sphx_glr_backref_{{ fullname }}:

   .. minigallery:: {{ fullname }}
       :add-heading:
