
## Ciclo de vída
###### ¿Cómo se gestionan los datos desde su generación hasta su eliminación en tu proyecto?
Los datos se gestionan atraves de ficheros intermedios de formato json, estos ficheros nunca serán borrados
a no ser que el usuario los borre personalmente ya que si se borran se estaría perdiendo todo el rastro de ese
proyecto.

##### ¿Qué estrategia sigues para garantizar la consistencia e integridad de los datos?
La estrategía que sigo para garantizar la consistencia e integridad es que he hecho que el código cuando
genere el fichero en el que almacena los datos lo cree de forma oculta y la integridad la aseguro 
mediante permisos, lo que hago es que solamente el usuario que haya ejecutado el programa podrá editar y borrar
dicho fichero. De esta forma si hay muchos usuarios en el mismo ordenador no podrían hacer nada a ese fichero.

##### Si no trabajas con datos, ¿cómo podrías incluir una funcionalidad que los gestione de forma eficiente?
Si trabajo con datos.

## Almacenamiento en la nube
##### Si tu software utiliza almacenamiento en la nube, ¿cómo garantizas la seguridad y disponibilidad de los datos?
No utilizo almacenamiento en la nube ya que lo veía muy complejo para este programa ya que tendría que introducir las credenciales de dicho almacenamiento en la nube y habilitar el programa para que acepte varios sistemas en la nube y eso es muy complejo para la funcionalidad que yo quiero hacer.

##### ¿Qué alternativas consideraste para almacenar datos y por qué elegiste tu solución actual?
La alternativa que considero para almacenar los datos son en ficheros temporales pero veía mejor esta opción ya que de esta forma el usuario podría cojer el fichero y llevarse a otra máquina para trabajar con el, mientras que si fuera un fichero temporal el usuario tendría que ponerse a buscar el fichero entre todos los temporales y quería que fuera útil no molesto.

##### Si no usas la nube, ¿cómo podrías integrarla en futuras versiones?
Podría integrarla para almacenar la estructura del json en la nube.

## Seguridad y regulación
##### ¿Qué medidas de seguridad implementaste para proteger los datos o procesos en tu proyecto?
La medida de seguridad que implementé son los permisos ya que solo el usuario que ejecutó ese programa podrá borrar y editar el fichero del programa, de esta forma nadie podría modificar el trabajo del usuario.
Estaba pensando en almacenar los datos encriptados pero si lo hiciera no tendría mucho sentido ya que el usuario que haya robado la estructura de otro usuario lo habría hecho para usarlo en este programa por lo que no tiene mucho sentido encriptar algo si cualquier es capaz de desencriptarlo. 

##### ¿Qué normativas (e.g., GDPR) podrían afectar el uso de tu software y cómo las has tenido en cuenta?

La normativa que podría afectar el uso de mi software sobre todo es el reglamento general de protección de datos
ya que como yo no encripto los datos no estaría cumpliendo las normativas de mi software.
También afectaría a mas normativas pero todas relacionadas con el uso de los datos y la privacidad del usuario.

###### Si no implementaste medidas de seguridad, ¿qué riesgos potenciales identificas y cómo los abordarías en el futuro?
El problema que tengo es que almaceno los datos sin encriptarlos pero si lo hiciera no tendría mucho sentido ya que el usuario que haya robado la estructura de otro usuario lo habría hecho para usarlo en este programa por lo que no tiene mucho sentido encriptar algo si cualquier es capaz de desencriptarlo. 
La medida de solución que se me ocurre es incluir una contraseña para que solo el usuario que puso la contraseña
pudiera desencriptarlo pero sinceramente para un programa para organizarte mejor no tiene mucho sentido tener tantas medidas de protección ante los datos ya que el programa lo que te ayuda es a estructurarte y a recordar funcionalidades del código.



## Implicación de las THD en negocio y planta (2e):
##### ¿Qué impacto tendría tu software en un entorno de negocio o en una planta industrial?
Tendría un impacto positivo ya que por ejemplo en un entorno de programación podrían analizar el software que tienen que hacer y podrían ir almacenandolo en el programa y podrían compartir el json a los trabajadores
y de esta forma todos tendrían la estructura final que quedó en ese momento cuando lo estaban abordando el software de esta forma no se les olvidaría ningun problema.



##### ¿Cómo crees que tu solución podría mejorar procesos operativos o la toma de decisiones?
Mi software mejoraria la toma de decisiones y el proceso operativo ya que mi programa permite centralizar el proyecto de una empresa y abordar todos los problemas de una forma muy clara y de forma que podrían abordar el proyecto más fácil. También los ayudaría ya que la herramienta de documentación y análisis harían que pudieran entender lo que hacen sus compañeros sin tenerse que leer todas las líneas de código o todos los cambios hechos en un repositorio. 


##### Si tu proyecto no aplica directamente a negocio o planta, ¿qué otros entornos podrían beneficiarse?
Si aplica a negocios.

## Mejoras en IT y OT
##### ¿Cómo puede tu software facilitar la integración entre entornos IT y OT?
No puede facilitar la incoporación en sistemas IT y OT a la vez. Pero si OT ya que podrían mejorar la colaboración entre los trabajadores, facilitar el análisis del código y promover la estandarización ya que estarían documentando todo.

##### ¿Qué procesos específicos podrían beneficiarse de tu solución en términos de automatización o eficiencia?
El proceso de análizar el código ofrecería una muy buena eficiencia a los usuarios y en automatización estaría el proceso de documentación ya que es automático y los usuarios no tienen que hacer nada para que lo haga y lo hace perfecto.

##### Si no aplica a IT u OT, ¿cómo podrías adaptarlo para mejorar procesos tecnológicos concretos?
Podría adaptarlo a para la optimización de procesos administrativos ya que podría modificar alguna de las funcionalidades para que fuera recopilando datos y poder realizar análisis a tiempo real de esos datos y predicir futuros posibles problemas y como solucionarlo. Por ejemplo: Una empresa de ropa tiene 1200 de stock de prendas de un chaleco y el programa a analizado eso y revisaría en internet y en toda la base de datos acerca de ese producto y por ejemplo si estamos en octubre y normalmente por més se compran 200 pero si son meses de veranos pues ahora como vendría invierno el programa te avisaría que tendrías que reponer el stock porque en invierno se vende mucho más.

## Tecnologías Habilitadoras Digitales
##### ¿Qué tecnologías habilitadoras digitales (THD) has utilizado o podrías integrar en tu proyecto?
Tecnologías utilizadas:
- `Inteligencia artificial`
- `Almacenamiento y gestión de datos`
Tecnologías que podría integrar:
- `Diagramas de clases uml`

Podría integrar diagramas de clases uml para explicar mejor el código.

##### ¿Cómo mejoran estas tecnologías la funcionalidad o el alcance de tu software?
Lo mejorarían en la funcionalidad de explicar el código proporcionando que el análisis de código sea muchisimo mejor que la version anterior. El alcance del software mejoraría un poco pero no mucho ya que realmente las funcionalidades que anteriores ya prácticamente te lo explicaban perfecto pero con el diagrama de clases podría aumentar
el alcanze del software.


##### Si no has utilizado THD, ¿cómo podrías implementarlas para enriquecer tu solución?
Si he utilizado THD