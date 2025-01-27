# Generated by Django 4.2 on 2025-01-25 15:37

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('models', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelFramwork',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('description', models.CharField(max_length=255)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'verbose_name_plural': 'Model Frameworks',
                'db_table': 'model_framwork',
            },
        ),
        migrations.CreateModel(
            name='ModelTask',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('description', models.CharField(max_length=255)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'verbose_name_plural': 'Model Tasks',
                'db_table': 'model_task',
            },
        ),
        migrations.AlterField(
            model_name='model',
            name='framework',
            field=models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to='models.modelframwork'),
        ),
        migrations.AlterField(
            model_name='model',
            name='task',
            field=models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to='models.modeltask'),
        ),
    ]
